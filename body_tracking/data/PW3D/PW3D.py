import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
# from config import cfg
from utils.human_models import smpl_x
from utils.preprocessing_body import load_img, process_bbox, augmentation, process_human_model_output
from utils.transforms import pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import render_mesh, vis_kp2d, save_obj, vis_bbox, vis_kp2d_bbox
import torchgeometry as tgm


class PW3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('../DATA-for-body-hand-training-2/data/PW3D', 'PW3D', 'data')

        # H36M joint set
        self.joint_set_h36m = {'body': \
                                   {'joint_num': 17,
                                    'joints_name': (
                                    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                                    'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder',
                                    'R_Elbow', 'R_Wrist'),
                                    'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                                    'smpl_regressor': np.load(osp.join('data', 'Human36M', 'J_regressor_h36m_smpl.npy'))
                                    }
                               }
        self.joint_set_h36m['body']['root_joint_idx'] = self.joint_set_h36m['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()
        print("------------PW3D dataset: ", len(self.datalist))

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 1
        elif self.data_split == 'test':
            return 10
        else:
            assert 0, print('Unknown subset')

    def load_data(self):
        db = COCO(osp.join(self.data_path, '3DPW_' + self.data_split + '.json'))
        with open(osp.join(self.data_path, '3DPW_' + self.data_split + '_SMPLX_NeuralAnnot.json')) as f:
            smplx_params = json.load(f)

        sampling_ratio = self.get_subsampling_ratio()

        datalist = []
        idx = 0
        for aid in db.anns.keys():
            idx += 1
            if idx % sampling_ratio != 0:
                continue

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            sequence_name = img['sequence']
            img_name = img['file_name']
            img_path = osp.join(self.data_path, 'imageFiles', sequence_name, img_name)
            cam_param = {k: np.array(v, dtype=np.float32) for k, v in img['cam_param'].items()}

            smplx_param = smplx_params[str(aid)]
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'], img['width']), 'bbox': bbox,
                         'smplx_param': smplx_param, 'cam_param': cam_param}
            datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape = data['img_path'], data['img_shape']

        # img
        img = load_img(img_path)
        bbox, smplx_param, cam_param = data['bbox'], data['smplx_param'], data['cam_param']
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        # smpl coordinates
        joint_img_ori, smplx_joint_img, smplx_joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
            smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
        smplx_joint_valid = np.ones((smpl_x.body_joint_num, 1), dtype=np.float32)
        pose_6d_valid = np.ones((smpl_x.orig_body_joint_num * 6), dtype=np.float32)
        pose_valid = np.ones((smpl_x.orig_body_joint_num * 3), dtype=np.float32)
        smplx_shape_valid = float(True)


        # convert from axis-angel to rotmat
        pose_6d = pose.copy().reshape(22, 3)  # (22,3)
        pose_6d = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(pose_6d))[:, 0:3, 0:2]  # tensor(22,3,2)
        pose_6d = np.array(pose_6d).reshape(22, -1)  # (22,6)
        pose_6d = pose_6d.reshape(-1)

        inputs = {'img': img}
        targets = {'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': pose,
                   'smplx_pose_6d': pose_6d, 'smplx_shape': shape, 'smplx_mesh_cam': mesh_cam_orig}
        meta_info = {'smplx_joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                     'pose_valid': pose_valid, 'pose_6d_valid': pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid, 'is_3D': float(True)}
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': [], 'pa_mpvpe': []}

        # 'Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3',
        # 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
        # 'R_Elbow', 'L_Wrist', 'R_Wrist',  # body joints
        joint_mapper = [1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]

        for n in range(sample_num):
            out = outs[n]

            # h36m joint from gt mesh
            mesh_gt_cam = out['smplx_mesh_cam_target'] #(10475,3)
            pose_coord_gt_h36m = np.dot(smpl_x.J_regressor, mesh_gt_cam) # (55,3)
            pose_coord_gt_h36m = pose_coord_gt_h36m[0:22, :]
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[0, None]  # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[joint_mapper, :] # (14,3)
            mesh_gt_cam -= np.dot(smpl_x.J_regressor, mesh_gt_cam)[0, None, :]

            # h36m joint from output mesh
            mesh_out_cam = out['smplx_mesh_cam']
            pose_coord_out_h36m = np.dot(smpl_x.J_regressor, mesh_out_cam) # (55,3)
            pose_coord_out_h36m = pose_coord_out_h36m[0:22, :]
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[0, None]  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[joint_mapper, :] #(14,3)
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m)
            eval_result['mpjpe'].append(
                np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(
                np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            mesh_out_cam -= np.dot(smpl_x.J_regressor, mesh_out_cam)[0, None, :]
            mesh_out_cam_aligned = rigid_align(mesh_out_cam, mesh_gt_cam)
            eval_result['mpvpe'].append(
                np.sqrt(np.sum((mesh_out_cam - mesh_gt_cam) ** 2, 1)).mean() * 1000)  # meter -> milimeter
            eval_result['pa_mpvpe'].append(
                np.sqrt(np.sum((mesh_out_cam_aligned - mesh_gt_cam) ** 2, 1)).mean() * 1000)  # meter -> milimeter

            vis = True
            if vis:
                file_name = str(cur_sample_idx + n)
                img = (out['img'].transpose(1, 2, 0)[:, :, ::-1] * 255).copy()
                # img = out['img']
                # img = img[:, :, ::-1].astype(np.uint8)
                # cv2.imshow("img", img)
                # cv2.waitKey(1)
                # cv2.imwrite('obj/'+file_name + '.jpg', img)
                # save_obj(mesh_gt_cam, smpl_x.face, 'obj/'+file_name + '_gt.obj')
                # save_obj(mesh_out_cam, smpl_x.face, 'obj/'+file_name + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))

        # f = open(os.path.join('train_infos3/body', 'result-b3.txt'), 'w')
        # f.write(f'3DPW-test dataset: \n')
        # f.write('MPJPE (Body): %.2f mm\n' % np.mean(eval_result['mpjpe_body']))
        # f.write('PA MPJPE (Body): %.2f mm\n' % np.mean(eval_result['pa_mpjpe_body']))
        #
        # f.write(f"{np.mean(eval_result['mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_body'])}")


# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    dsplit = 'test'
    pw3d = PW3D(transform=None, data_split=dsplit)
    datalist = pw3d.datalist
    cv2.namedWindow('img-kp2d', 0)
    cv2.namedWindow('smplx-overlay', 0)
    cv2.namedWindow('img-box', 0)

    for idx in range(len(datalist)):
        data = copy.deepcopy(datalist[idx])
        img_path, img_shape, bbox, cam_param, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data[
            'cam_param'], data['smplx_param']
        print(img_path)

        img = cv2.imread(img_path)
        img2 = vis_bbox(img, bbox)
        cv2.imshow('img-box', img2)

        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)
        cv2.imshow('cropped-body-img', img[:, :, ::-1].astype(np.uint8))

        # joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
        #     smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
        joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
            smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

        # show ori img
        img = cv2.imread(img_path)
        # img2 = vis_kp2d(img, joint_img_ori)
        img2 = vis_kp2d_bbox(img, joint_img_ori, bbox)
        # cv2.imwrite(str(idx)+'.jpg', img2)
        cv2.imshow('img-kp2d', img2)

        # mesh render
        img = cv2.imread(img_path)
        rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
        # cv2.imwrite(str(idx)+'.jpg', rendered_img)
        cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

        cv2.waitKey(1)
    pass
