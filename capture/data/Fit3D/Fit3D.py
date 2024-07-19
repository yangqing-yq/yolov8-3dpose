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
from utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output_new, process_human_model_output_new_debug
from utils.transforms import world2cam, cam2pixel, rigid_align
from utils.vis import vis_kp2d, render_mesh, save_obj, vis_bbox, vis_kp2d_bbox
import torchgeometry as tgm

class Fit3D(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, configs=None):
        self.transform = transform
        self.data_split = data_split
        # self.cfg = configs
        self.img_dir = osp.join('../DATA', 'Fit3D')
        self.annot_path = osp.join('../DATA', 'Fit3D', 'annotations')
        # # H36M joint set
        # self.joint_set = {'joint_num': 17,
        #                 'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
        #                 'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
        #                 'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
        #                 'regressor': np.load(osp.join('data', 'Human36M', 'J_regressor_h36m_smplx.npy'))
        #                 }
        # self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')


        self.datalist = self.load_data()
        print("------------Fit3D dataset: ", len(self.datalist))

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 10
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [3]#[3,4,5,7,8,9]
        elif self.data_split == 'test':
            subject = [10] #[10,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smplx_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Fit3D_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Fit3D_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            # with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
            #     joints[str(subject)] = json.load(f)
            # smplx parameter load
            with open(osp.join(self.annot_path, 'Fit3D_subject' + str(subject) + '_smplx_params.json'),'r') as f:
                smplx_params[str(subject)] = json.load(f)

        db.createIndex()

        datalist = []
        idx = 0
        for aid in db.anns.keys():
            idx += 1
            if idx % sampling_ratio != 0:
                continue

            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # # check subject and frame_idx
            # frame_idx = img['frame_idx']
            # if frame_idx % sampling_ratio != 0:
            #     continue

            # smplx parameter
            subject = img['subject']; cam_idx = img['cam_idx']; action_idx = img['action']; frame_idx = img['frame_idx']; 
            smplx_param = {
                'root_pose': np.array(smplx_params[str(subject)][action_idx]['global_orient'][frame_idx]),
                'body_pose': np.array(smplx_params[str(subject)][action_idx]['body_pose'][frame_idx]),
                'shape': np.array(smplx_params[str(subject)][action_idx]['betas'][frame_idx]),
                'trans': np.array(smplx_params[str(subject)][action_idx]['transl'][frame_idx]),
                'lhand_pose': np.array(smplx_params[str(subject)][action_idx]['left_hand_pose'][frame_idx]),
                'rhand_pose': np.array(smplx_params[str(subject)][action_idx]['right_hand_pose'][frame_idx]),
                'jaw_pose': np.array(smplx_params[str(subject)][action_idx]['jaw_pose'][frame_idx]),
                'leye_pose': np.array(smplx_params[str(subject)][action_idx]['leye_pose'][frame_idx]),
                'reye_pose': np.array(smplx_params[str(subject)][action_idx]['reye_pose'][frame_idx]),
                'expression': np.array(smplx_params[str(subject)][action_idx]['expression'][frame_idx]),
            }

            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}

            # -------------update smplx params(root_pose & tans) according to cam_param extrinsic: mesh_cam = np.matmul(mesh_cam - t, R.transpose())
            # update root_pose
            root_pose = smplx_param['root_pose']
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = root_pose.reshape(1, 3)
            smplx_param['root_pose'] = root_pose
            # compute pelvis, update trans
            with torch.no_grad():
                torch_shape = torch.FloatTensor(smplx_param['shape']).view(1, -1)
                output = smpl_x.layer['neutral'](betas=torch_shape, body_pose=torch.zeros((1, 63)).float(), global_orient=torch.zeros((1, 3)).float(),
                                          transl=torch.zeros((1, 3)).float(), left_hand_pose=torch.zeros((1, 45)).float(),
                                          right_hand_pose=torch.zeros((1, 45)).float(), jaw_pose=torch.zeros((1, 3)).float(),
                                          leye_pose=torch.zeros((1, 3)).float(), reye_pose=torch.zeros((1, 3)).float(), expression=torch.zeros((1, 10)).float())
                joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :]  # (137,3)
                pelvis = joint_cam[0:1, :]
            t = t.reshape(1, 3)
            trans = smplx_param['trans']
            trans = np.dot(R, (trans - t).transpose(1, 0)).transpose(1, 0)
            delta = np.dot(R, pelvis.transpose(1, 0)) - pelvis.transpose(1, 0)
            trans = trans + delta.transpose(1, 0)
            smplx_param['trans'] = trans
        
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'smplx_param': smplx_param,
                'cam_param': cam_param})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
        
        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
            
        # smplx parameters
        smplx_param = data['smplx_param']
        smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_mesh_cam_orig, _, _ = process_human_model_output_new(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
        smplx_joint_valid = np.ones((smpl_x.body_joint_num, 1), dtype=np.float32)
        smplx_pose_valid = np.ones((smpl_x.orig_body_joint_num*3), dtype=np.float32)
        smplx_pose_6d_valid = np.ones((smpl_x.orig_body_joint_num*6), dtype=np.float32)
        smplx_shape_valid = float(True)

        # convert from axis-angel to rotmat
        smplx_pose_6d = smplx_pose.copy().reshape(22, 3)  # (22,3)
        smplx_pose_6d = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(smplx_pose_6d))[:, 0:3, 0:2]  # tensor(22,3,2)
        smplx_pose_6d = np.array(smplx_pose_6d).reshape(22, -1) # (22,6)
        smplx_pose_6d = smplx_pose_6d.reshape(-1)


        inputs = {'img': img}
        targets = {'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 'smplx_pose_6d': smplx_pose_6d, 'smplx_mesh_cam': smplx_mesh_cam_orig}
        meta_info = {'smplx_joint_trunc': smplx_joint_trunc, 'smplx_joint_valid': smplx_joint_valid, 'smplx_pose_valid': smplx_pose_valid, 'smplx_pose_6d_valid': smplx_pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid, 'is_3D': float(True)}
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
            mesh_gt_cam = out['smplx_mesh_cam_target']  # (10475,3)
            pose_coord_gt_h36m = np.dot(smpl_x.J_regressor, mesh_gt_cam)  # (55,3)
            pose_coord_gt_h36m = pose_coord_gt_h36m[0:22, :]
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[0, None]  # root-relative
            pose_coord_gt_h36m = pose_coord_gt_h36m[joint_mapper, :]  # (14,3)
            mesh_gt_cam -= np.dot(smpl_x.J_regressor, mesh_gt_cam)[0, None, :]

            # h36m joint from output mesh
            mesh_out_cam = out['smplx_mesh_cam']
            mesh_out_cam_pred = mesh_out_cam.copy()
            pose_coord_out_h36m = np.dot(smpl_x.J_regressor, mesh_out_cam)  # (55,3)
            pose_coord_out_h36m = pose_coord_out_h36m[0:22, :]
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[0, None]  # root-relative
            pose_coord_out_h36m = pose_coord_out_h36m[joint_mapper, :]  # (14,3)
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

            # image show evaluation result
            vis = True
            if vis:
                img = (out['img'].transpose(1, 2, 0)[:, :, ::-1] * 255).copy()
                img = img.astype(np.uint8)

                cam_param = {}
                cam_param['focal'] = (5000, 5000)
                cam_param['princpt'] = (192 / 2, 256 / 2)
                rendered_img = render_mesh(img, mesh_out_cam_pred, smpl_x.face, cam_param)
                cv2.imshow("img", rendered_img.astype(np.uint8))
                cv2.waitKey(0)
                # cv2.imwrite('obj/'+file_name + '.jpg', img)
                # save_obj(mesh_gt_cam, smpl_x.face, 'obj/'+file_name + '_gt.obj')
                # save_obj(mesh_out_cam, smpl_x.face, 'obj/'+file_name + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))



# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    dsplit = 'train'
    fit3d = Fit3D(transform=None, data_split=dsplit)
    datalist = fit3d.datalist
    cv2.namedWindow('img-kp2d-box', 0)
    cv2.namedWindow('smplx-overlay-ori', 0)

    for idx in range(len(datalist)):
        data = copy.deepcopy(datalist[idx])
        img_path, img_shape, bbox, cam_param, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data[
            'cam_param'], data['smplx_param']
        print(img_path)

        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)

        joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig, mesh_cam_aug, joint_img_ori, joint_img_crop = process_human_model_output_new_debug(
            smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

        # show ori img
        img0 = cv2.imread(img_path)
        img2 = vis_kp2d_bbox(img0, joint_img_ori, bbox)
        # cv2.imwrite(str(idx)+'.jpg', img2)
        cv2.imshow('img-kp2d-box', img2)

        # mesh render on ori img
        rendered_img = render_mesh(img0, mesh_cam_orig, smpl_x.face, cam_param)
        # cv2.imwrite(str(idx)+'.jpg', rendered_img)
        cv2.imshow('smplx-overlay-ori', rendered_img.astype(np.uint8))

        # show cropped img
        img2 = img[:, :, ::-1].astype(np.uint8)
        img3 = vis_kp2d(img2, joint_img_crop)
        cv2.imshow('cropped-img-kp2d', img3)

        # mesh render on cropped img
        am_param = {}
        cam_param['focal'] = (5000, 5000)
        cam_param['princpt'] = (192 / 2, 256 / 2)
        rendered_img = render_mesh(img2, mesh_cam_aug, smpl_x.face, cam_param)
        cv2.imshow('smplx-overlay-aug', rendered_img.astype(np.uint8))

        cv2.waitKey(0)
    pass