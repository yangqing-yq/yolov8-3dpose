import os
import os.path as osp
import numpy as np
# from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from utils.human_models import smpl, mano, flame
from utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.vis import vis_keypoints, vis_mesh, save_obj

class MSCOCO(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('../body-hand-training/data/MSCOCO', 'MSCOCO', 'images')
        self.annot_path = osp.join('../body-hand-training/data/MSCOCO', 'MSCOCO', 'annotations')
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
            with open(osp.join(self.annot_path, 'MSCOCO_train_SMPL_NeuralAnnot.json')) as f:
                smpl_params = json.load(f)
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

                smpl_param = smpl_params[str(aid)]

                data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'smpl_param': smpl_param} 
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
    
            # coco gt
            dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
            joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

            # smpl fitted data
            smpl_param = data['smpl_param']
            smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(smpl_param['smpl_param'], smpl_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'smpl')
            smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
            smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
            smpl_shape_valid = float(True)

            """
            # for debug
            _tmp = smpl_joint_img.copy() 
            _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
            _img = vis_keypoints(_img, _tmp)
            cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
            """
                
            inputs = {'img': img}
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smpl_joint_img': smpl_joint_img, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(False)}
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
