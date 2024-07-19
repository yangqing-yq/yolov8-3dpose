import os
import os.path as osp
import numpy as np
# from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl_x
from utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.vis import render_mesh, vis_kp2d
import torchgeometry as tgm

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('../DATA-for-body-hand-training-2/data/MSCOCO', 'MSCOCO', 'images')
        self.annot_path = osp.join('../DATA-for-body-hand-training-2/data/MSCOCO', 'MSCOCO', 'annotations')
        # self.img_path = osp.join('/datahdd/wyz/workspace/full_body_mocap/train/Hand4Whole/data/MSCOCO', 'images')
        # self.annot_path = osp.join('/datahdd/wyz/workspace/full_body_mocap/train/Hand4Whole/data/MSCOCO', 'annotations')
        
        # mscoco joint set
        self.joint_set = {'body': \
                            {'joint_num': 32, 
                            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Index_1', 'L_Middle_1', 'L_Ring_1', 'L_Pinky_1', 'R_Index_1', 'R_Middle_1', 'R_Ring_1', 'R_Pinky_1'),
                            'flip_pairs': ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) , (18, 21), (19, 22), (20, 23), (24, 28), (25, 29) ,(26, 30), (27, 31) )
                            },\
                    'hand': \
                            {'joint_num': 21,
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            },
                    'face': \
                            {
                            'joint_to_flame': (-1, -1, -1, -1, -1, # no joints for neck, backheads, eyeballs
                                            17, 18, 19, 20, 21, # right eyebrow
                                            22, 23, 24, 25, 26, # left eyebrow
                                            27, 28, 29, 30, # nose
                                            31, 32, 33, 34, 35, # below nose
                                            36, 37, 38, 39, 40, 41, # right eye
                                            42, 43, 44, 45, 46, 47, # left eye
                                            48, # right lip
                                            49, 50, 51, 52, 53, # top lip
                                            54, # left lip
                                            55, 56, 57, 58, 59, # down lip
                                            60, 61, 62, 63, 64, 65, 66, 67, # inside of lip
                                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 # face contour
                                            )
                            }
                        }
        self.datalist = self.load_data()
        print("------------MSCOCO dataset: ", len(self.datalist))
    
    def add_joint(self, joint_coord, feet_joint_coord, ljoint_coord, rjoint_coord):
        # pelvis
        lhip_idx = self.joint_set['body']['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['body']['joints_name'].index('R_Hip')
        pelvis = (joint_coord[lhip_idx,:] + joint_coord[rhip_idx,:]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
        pelvis = pelvis.reshape(1,3)
        
        # feet
        lfoot = feet_joint_coord[:3,:]
        rfoot = feet_joint_coord[3:,:]
        
        # hands
        lhand = ljoint_coord[[5,9,13,17], :]
        rhand = rjoint_coord[[5,9,13,17], :]

        joint_coord = np.concatenate((joint_coord, pelvis, lfoot, rfoot, lhand, rhand)).astype(np.float32)
        return joint_coord

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            with open(osp.join(self.annot_path, 'MSCOCO_train_SMPLX_all_NeuralAnnot.json')) as f:
                smplx_params = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))


        # train mode
        if self.data_split == 'train':
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                # body part
                if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                    continue
                
                # bbox
                bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
                if bbox is None: continue
                
                # joint coordinates
                joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                foot_joint_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1,3)
                ljoint_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3)
                rjoint_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)
                joint_img = self.add_joint(joint_img, foot_joint_img, ljoint_img, rjoint_img)
                joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                joint_img[:,2] = 0

                smplx_param = smplx_params[str(aid)]

                data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'smplx_param': smplx_param}
                datalist.append(data_dict)

            return datalist

        # test mode
        else:
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
                if bbox is None: continue

                data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox}
                datalist.append(data_dict)
            return datalist
 
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # train mode
        if self.data_split == 'train':
            img_path, img_shape = data['img_path'], data['img_shape']
            
            # image load
            img = load_img(img_path)
            
            # body part
            # affine transform
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32))/255.
    
            # # coco gt
            # dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
            # joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

            # smpl fitted data
            smplx_param = data['smplx_param']
            cam_param = smplx_param['cam_param']
            joint_img_ori, smplx_joint_img, smplx_joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
                smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            smplx_joint_valid = np.ones((smpl_x.body_joint_num, 1), dtype=np.float32)
            pose_valid = np.ones((smpl_x.orig_body_joint_num * 3), dtype=np.float32)
            pose_6d_valid = np.ones((smpl_x.orig_body_joint_num * 6), dtype=np.float32)
            smplx_shape_valid = float(True)

            # convert from axis-angel to rotmat
            pose_6d = pose.copy().reshape(22, 3)  # (22,3)
            pose_6d = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(pose_6d))[:, 0:3, 0:2]  # tensor(22,3,2)
            pose_6d = np.array(pose_6d).reshape(22, -1)  # (22,6)
            pose_6d = pose_6d.reshape(-1)

            inputs = {'img': img}
            targets = {'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': pose, 'smplx_pose_6d': pose_6d,
                       'smplx_shape': shape, 'smplx_mesh_cam':mesh_cam_orig}
            meta_info = {'smplx_joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                         'pose_valid': pose_valid, 'pose_6d_valid': pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid,
                         'is_3D': float(True)}
            return inputs, targets, meta_info

        # test mode
        else:
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)

            # body part
            # affine transform
            bbox = data['bbox']
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32))/255.
            
            inputs = {'img': img}
            targets = {}
            meta_info = {'bb2img_trans': bb2img_trans}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]
        return {}

    def print_eval_result(self, eval_result):
        return


# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    dsplit = 'train'
    coco = MSCOCO(transform=None, data_split=dsplit)
    datalist = coco.datalist
    cv2.namedWindow('img-kp2d',0)
    cv2.namedWindow('smplx-overlay', 0)

    for idx in range(len(datalist)):
        data = copy.deepcopy(datalist[idx])
        img_path, img_shape, bbox, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data['smplx_param']
        cam_param = smplx_param['cam_param']

        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)
        cv2.imshow('cropped-body-img', img[:,:,::-1].astype(np.uint8))

        joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
            smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

        # show ori img
        img = cv2.imread(img_path)
        img2 = vis_kp2d(img, joint_img_ori)
        cv2.imshow('img-kp2d', img2)

        # mesh render
        img = cv2.imread(img_path)
        rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
        # cv2.imwrite(str(idx)+'.jpg', rendered_img)
        cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

        cv2.waitKey(0)
    pass