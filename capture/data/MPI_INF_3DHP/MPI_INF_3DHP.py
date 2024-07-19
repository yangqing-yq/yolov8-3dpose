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
from utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import world2cam, cam2pixel
from utils.vis import render_mesh, vis_kp2d, save_obj, vis_bbox, vis_kp2d_bbox
import torchgeometry as tgm

input_img_shape = (256, 192)
output_hm_shape = (8, 8, 6)

class MPI_INF_3DHP(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join('../DATA-for-body-hand-training-2/data/MPI_INF_3DHP', 'MPI_INF_3DHP', 'data')

        
        # MPI-INF-3DHP joint set
        self.joint_set = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('Head_top', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Head'),
                            'flip_pairs': ( (2,5), (3,6), (4,7), (8,11), (9,12), (10,13) )
                            }
                        }
        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()
        print("------------MPI_INF_3DHP dataset: ", len(self.datalist))
        
    def load_data(self):
        db = COCO(osp.join(self.data_path, 'MPI-INF-3DHP_1k.json'))
        with open(osp.join(self.data_path, 'MPI-INF-3DHP_joint_3d.json')) as f:
            joints = json.load(f)
        with open(osp.join(self.data_path, 'MPI-INF-3DHP_camera_1k.json')) as f:
            cameras = json.load(f)
        # smpl parameters load
        smplx_param_path = osp.join(self.data_path, 'MPI-INF-3DHP_SMPLX_NeuralAnnot.json')
        with open(smplx_param_path,'r') as f:
            smplx_params = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            subject_idx = img['subject_idx']
            seq_idx = img['seq_idx']
            frame_idx = img['frame_idx']
            img_path = osp.join(self.data_path, 'images_1k', 'S' + str(subject_idx), 'Seq' + str(seq_idx), 'imageSequence', img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # frame sampling (25 frame per sec -> 25/3 frame per sec)
            if frame_idx % 6 != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject_idx)][str(seq_idx)][str(cam_idx)]
            R, t, focal, princpt = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal':focal, 'princpt':princpt}
            
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject_idx)][str(seq_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, focal, princpt)
            joint_valid = np.ones_like(joint_img[:,:1])

            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue

            # smpl parameter
            smplx_param = smplx_params[str(subject_idx)][str(seq_idx)][str(frame_idx)]
    
            datalist.append({
                'img_path': img_path,
                'img_id': image_id,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'cam_param': cam_param,
                'smplx_param': smplx_param})

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, smplx_param, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['smplx_param'], data['cam_param']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
         
        # # mi gt
        # joint_cam = data['joint_cam']
        # joint_cam = (joint_cam - joint_cam[self.joint_set['body']['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
        # joint_img = data['joint_img']
        # joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
        # joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)
        #
        # smpl coordinates and parameters
        cam_param['t'] /= 1000 # milimeter to meter
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
        meta_info = {'smplx_joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid, 'pose_valid': pose_valid,
                     'pose_6d_valid': pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid, 'is_3D': float(True)}

        """
        # for debug
        _tmp = smplx_joint_img.copy()
        _tmp[:,0] = _tmp[:,0] / output_hm_shape[2] * input_img_shape[1]
        _tmp[:,1] = _tmp[:,1] / output_hm_shape[1] * input_img_shape[0]
        _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
        _img = vis_kp2d(_img.copy(), _tmp)
        cv2.imshow('smplx_joint_img_kp2d', _img.astype(np.uint8))
        cv2.waitKey(1)
        # cv2.imwrite('mpii_' + str(idx) + '.jpg', _img)
        """


        return inputs, targets, meta_info


# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    dsplit = 'test'
    mpi_inf_3dhp = MPI_INF_3DHP(transform=None, data_split=dsplit)
    datalist = mpi_inf_3dhp.datalist
    cv2.namedWindow('img-kp2d',0)
    cv2.namedWindow('smplx-overlay', 0)

    for idx in range(len(datalist)):
        data = copy.deepcopy(datalist[idx])
        img_path, img_shape, bbox, cam_param, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param'], data['smplx_param']


        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)
        cv2.imshow('cropped-body-img', img[:,:,::-1].astype(np.uint8))

        cam_param['t'] /= 1000  # milimeter to meter
        # joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
        #     smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
        joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
            smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

        # show ori img
        img = cv2.imread(img_path)
        # img2 = vis_kp2d(img, joint_img_ori)
        img2 = vis_kp2d_bbox(img, joint_img_ori, bbox)
        cv2.imshow('img-kp2d', img2)

        # mesh render
        img = cv2.imread(img_path)
        rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
        # cv2.imwrite(str(idx)+'.jpg', rendered_img)
        cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

        cv2.waitKey(0)
    pass