import os
import os.path as osp
import numpy as np
# from config import cfg
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
import sys
sys.path.insert(1, '/dc/yolo3dpose/mod_sifan/3dat-tracking/body-tracking')
from body_tracking.utils.human_models import smpl_x
from body_tracking.utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from body_tracking.utils.vis import render_mesh, vis_kp2d, vis_kp2d_bbox
import torchgeometry as tgm

input_img_shape = (256, 192)
output_hm_shape = (8, 8, 6)

class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('/data/coco_human/images/train', 'train_ubody')
        self.annot_path = osp.join('/data/ubody_annotations/Movie')
        
        # mpii skeleton
        self.joint_set = {'body':
                            {'joint_num': 16,
                            'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
                            'flip_pairs': ( (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13) )
                            }
                        }
        self.datalist = self.load_data()
        print("------------MPII dataset: ", len(self.datalist))

    def process_imgname(self, imgname):
        parts = imgname.split('/')
        # modified_parts = [part.replace('_', '-') for part in parts]
        processed_imgname = '-'.join(parts)
        return processed_imgname


    def load_data(self):
        db = COCO(osp.join(self.annot_path, 'keypoint_annotation.json'))
        with open(osp.join(self.annot_path, 'smplx_annotation.json')) as f:
            smplx_params = json.load(f)
        # print("smplx_params",smplx_params)
        datalist = []
        for aid in db.anns.keys():
            # print("aid：",aid)
            ann = db.anns[aid]
            # print("ann",len(ann))
            # print("ann",ann)
            # print("ann['image_id']",ann['image_id'])
  
            img = db.loadImgs(ann['image_id'])[0]
            imgname = img['file_name']
            # print("imgname:",imgname)
            self.dir_name = self.annot_path.split("/")[-1]
            img_path = self.dir_name + '-' + self.process_imgname(imgname)
            # print("img_path1:",img_path)      
            img_path = osp.join(self.img_path, img_path)
            # print("img_path2:",img_path)
            if not os.path.exists(img_path):
                self.img_path = self.img_path.replace("train", "val")
                img_path = osp.join(self.img_path, imgname)
            if not os.path.exists(img_path):
                continue

            # bbox
            bbox = process_bbox(ann['bbox'], img['width'], img['height']) 
            if bbox is None: continue
            
            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
            joint_valid = joint_img[:,2:].copy()
            joint_img[:,2] = 0

            try:
                smplx_param = smplx_params[str(aid)]
            except Exception as e:
                continue

            datalist.append({
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param
            })
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # image load and affine transform
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # # mpii gt
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
        targets = {'smplx_joint_img': smplx_joint_img, 'smplx_joint_cam': smplx_joint_cam, 'smplx_pose': pose,
                   'smplx_pose_6d': pose_6d,
                   'smplx_shape': shape, 'smplx_mesh_cam': mesh_cam_orig}
        meta_info = {'smplx_joint_trunc': joint_trunc, 'smplx_joint_valid': smplx_joint_valid,
                     'pose_valid': pose_valid, 'pose_6d_valid': pose_6d_valid, 'smplx_shape_valid': smplx_shape_valid,
                     'is_3D': float(True)}

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
    dsplit = 'train'
    mpii = MPII(transform=None, data_split=dsplit)
    datalist = mpii.datalist
    cv2.namedWindow('img-kp2d',0)
    cv2.namedWindow('smplx-overlay', 0)

    # print("datalist:",datalist)


    idx=100
    data = copy.deepcopy(datalist[idx])
    img_path, img_shape, bbox, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data['smplx_param']
    print("img_path:",img_path )
    print("img_shape:", img_shape )
    print("bbox:", bbox)
    print("smplx_param:",smplx_param)
    cam_param = smplx_param['cam_param']

    img = load_img(img_path)
    img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)
    cv2.imshow('cropped-body-img', img[:,:,::-1].astype(np.uint8))

    # anno
    joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
        smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', None)
    print("joint_img_ori:",joint_img_ori)
    print("joint_img:",joint_img)
    print("pose:",pose)
    print("shape:",shape)
    print("mesh_cam_orig:",mesh_cam_orig)
    print("joint_cam:",joint_cam)
    print("joint_trunc:",joint_trunc)


    # pred
    with open('pred_val.json', 'r') as f:
        pred_json = json.load(f)
    print("pred_json:",pred_json)
    pred_anno = pred_json
    joint_img_ori_pred, joint_img_pred, joint_cam_pred, joint_trunc_pred, pose_pred, shape_pred, mesh_cam_orig_pred = process_human_model_output(
        smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', pred_anno)

    # show ori img
    img = cv2.imread(img_path)
    img2 = vis_kp2d_bbox(img, joint_img_ori, bbox)
    cv2.imshow('img-kp2d', img2)

    # mesh render
    rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
    # cv2.imwrite(str(idx)+'.jpg', rendered_img)
    cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

    rendered_predimg = render_mesh(img, mesh_cam_orig_pred, smpl_x.face, cam_param)
    h, w = rendered_img.shape[:2]
    rendered_predimg = cv2.resize(rendered_predimg, (int(w * 0.3), int(h * 0.3)))
    cv2.imshow('smplx-overlay_pred', rendered_predimg.astype(np.uint8))

    cv2.waitKey(0)

    # for idx in range(len(datalist)):
    #     print("idx:",idx)
    #     data = copy.deepcopy(datalist[idx])
    #     img_path, img_shape, bbox, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data['smplx_param']
    #     cam_param = smplx_param['cam_param']

    #     try:
    #         pred_anno = pred_json[img_path.split("/")[-1]]
    #         if len(pred_anno['root_pose']) == 0:
    #             continue
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         continue

    #     img = load_img(img_path)
    #     img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)
    #     cv2.imshow('cropped-body-img', img[:,:,::-1].astype(np.uint8))

    #     # anno
    #     joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
    #         smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', None)

    #     # pred
    #     joint_img_ori_pred, joint_img_pred, joint_cam_pred, joint_trunc_pred, pose_pred, shape_pred, mesh_cam_orig_pred = process_human_model_output(
    #         smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', pred_anno)

    #     # show ori img
    #     img = cv2.imread(img_path)
    #     img2 = vis_kp2d_bbox(img, joint_img_ori, bbox)
    #     cv2.imshow('img-kp2d', img2)

    #     # mesh render
    #     rendered_img = render_mesh(img, mesh_cam_orig, smpl_x.face, cam_param)
    #     # cv2.imwrite(str(idx)+'.jpg', rendered_img)
    #     cv2.imshow('smplx-overlay', rendered_img.astype(np.uint8))

    #     rendered_predimg = render_mesh(img, mesh_cam_orig_pred, smpl_x.face, cam_param)
    #     h, w = rendered_img.shape[:2]
    #     rendered_predimg = cv2.resize(rendered_predimg, (int(w * 0.3), int(h * 0.3)))
    #     cv2.imshow('smplx-overlay_pred', rendered_predimg.astype(np.uint8))

    #     cv2.waitKey(0)
    # pass