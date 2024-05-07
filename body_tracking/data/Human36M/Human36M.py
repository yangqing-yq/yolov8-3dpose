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
from utils.transforms import world2cam, cam2pixel, rigid_align
from utils.vis import render_mesh, vis_kp2d
import torchgeometry as tgm

class Human36M(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('../DATA-for-body-hand-training-2/data/Human36M', 'Human36M', 'images')
        self.annot_path = osp.join('../DATA-for-body-hand-training-2/data/Human36M', 'Human36M', 'annotations')
        # self.img_dir = '/datahdd/wyz/workspace/full_body_mocap/train/Hand4Whole/data/Human36M/images'
        # self.annot_path = '/datahdd/wyz/workspace/full_body_mocap/train/Hand4Whole/data/Human36M/annotations'
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        # H36M joint set
        self.joint_set = {'body': \
                            {'joint_num': 17,
                            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
                            'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
                            'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                            'smpl_regressor': np.load(osp.join('data', 'Human36M', 'J_regressor_h36m_smpl.npy')),
                            }
                        }
        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')


        self.datalist = self.load_data()
        print("------------Human36M dataset: ", len(self.datalist))
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 10
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
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
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPLX_NeuralAnnot.json'),'r') as f:
                smplx_params[str(subject)] = json.load(f)

        db.createIndex()

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # smpl parameter
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            smplx_param = smplx_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]

            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)[:,:2]
            joint_valid = np.ones((self.joint_set['body']['joint_num'],1))
        
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
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
        
        if self.data_split == 'train':
            # # h36m gt
            # joint_cam = data['joint_cam']
            # joint_cam = (joint_cam - joint_cam[self.joint_set['body']['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
            # joint_img = data['joint_img']
            # joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)
            # joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)
            
            smplx_param = data['smplx_param']
            # smpl coordinates and parameters
            cam_param['t'] /= 1000 # milimeter to meter
            joint_img_ori, smplx_joint_img, smplx_joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')
            smplx_joint_valid = np.ones((smpl_x.body_joint_num, 1), dtype=np.float32)
            pose_6d_valid = np.ones((smpl_x.orig_body_joint_num*6), dtype=np.float32)
            pose_valid = np.ones((smpl_x.orig_body_joint_num * 3), dtype=np.float32)
            smplx_shape_valid = float(True)

            # convert from axis-angel to rotmat
            pose_6d = pose.copy().reshape(22, 3)  # (22,3)
            pose_6d = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(pose_6d))[:, 0:3, 0:2]  # tensor(22,3,2)
            pose_6d = np.array(pose_6d).reshape(22, -1) # (22,6)
            pose_6d = pose_6d.reshape(-1)

            inputs = {'img': img}
            targets = {'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': pose, 'smplx_pose_6d': pose_6d, 'smplx_shape': shape, 'smplx_mesh_cam':mesh_cam_orig}
            meta_info = {'smplx_joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid, 'pose_valid': pose_valid, 'pose_6d_valid': pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid, 'is_3D': float(True)}
            return inputs, targets, meta_info
        else:
            inputs = {'img': img}
            targets = {}
            meta_info = {'bbox': bbox}
            return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': []}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]
            
            # h36m joint from gt mesh
            joint_gt = annot['joint_cam'] 
            joint_gt = joint_gt - joint_gt[self.joint_set['body']['root_joint_idx'],None] # root-relative 
            joint_gt = joint_gt[self.joint_set['body']['eval_joint'],:] 
            
            # h36m joint from param mesh
            mesh_out = out['smpl_mesh_cam'] * 1000 # meter to milimeter
            joint_out = np.dot(self.joint_set['body']['smpl_regressor'], mesh_out) # meter to milimeter
            joint_out = joint_out - joint_out[self.joint_set['body']['root_joint_idx'],None] # root-relative
            joint_out = joint_out[self.joint_set['body']['eval_joint'],:]
            joint_out_aligned = rigid_align(joint_out, joint_gt)
            eval_result['mpjpe'].append(np.sqrt(np.sum((joint_out - joint_gt)**2,1)).mean())
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((joint_out_aligned - joint_gt)**2,1)).mean())

            # vis = False
            # if vis:
            #     filename = annot['img_path'].split('/')[-1][:-4]

            #     img = load_img(annot['img_path'])[:,:,::-1]
            #     img = vis_mesh(img, mesh_out_img, 0.5)
            #     cv2.imwrite(filename + '.jpg', img)
            #     save_obj(mesh_out, smpl.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))



# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    dsplit = 'test'
    hm36 = Human36M(transform=None, data_split=dsplit)
    datalist = hm36.datalist
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
        img2 = vis_kp2d(img, joint_img_ori)
        cv2.imshow('img-kp2d', img2)

        # mesh render
        img = cv2.imread(img_path)
        rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
        # cv2.imwrite(str(idx)+'.jpg', rendered_img)
        cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

        cv2.waitKey(0)
    pass