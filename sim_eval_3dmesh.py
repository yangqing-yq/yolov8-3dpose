import os
import os.path as osp
import numpy as np
# from config import cfg
import csv
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
import sys
sys.path.insert(1, '/dc/yolo3dpose/yolov8-3dpose')
from body_tracking.utils.human_models import smpl_x
from body_tracking.utils.preprocessing_body import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from body_tracking.utils.vis import render_mesh, vis_kp2d, vis_kp2d_bbox
from body_tracking.utils.transforms import pixel2cam, rigid_align, transform_joint_to_other_db
import torchgeometry as tgm

root_dir = os.path.dirname(os.path.dirname(__file__))
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
if __package__:
    from .infer.det_api import DetBodyHandHead
else:
    from infer.det_api import DetBodyHandHead
sys.path.pop()


input_img_shape = (256, 192)
output_hm_shape = (8, 8, 6)

class MPII(torch.utils.data.Dataset):
    def __init__(self, transform, data_split, subdir):
        self.subdir_name = subdir
        self.transform = transform
        self.data_split = data_split
        self.img_path = osp.join('/data/coco_human/images',self.data_split,self.data_split+'_ubody')
        
        self.annot_path = osp.join('/data/ubody_annotations/',self.subdir_name)
        
        # mpii skeleton
        self.joint_set_h36m = {'body':
                            {'joint_num': 16,
                            'joints_name': ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Thorax', 'Neck', 'Head_top', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'),
                            'flip_pairs': ( (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13) )
                            }
                        }
        
        self.joint_set_h36m['body']['root_joint_idx'] = self.joint_set_h36m['body']['joints_name'].index('Pelvis')

        self.datalist = self.load_data()
        print("---MPII dataset---: ",self.subdir_name,'subset ,samples:', len(self.datalist))

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

        # print("targets:",targets)
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
            mesh_out_cam_pred = mesh_out_cam.copy()
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

            # vis = True
            # if vis:
            #     file_name = str(cur_sample_idx + n)
            #     img = (out['img'].transpose(1, 2, 0)[:, :, ::-1] * 255).copy()
            #     img = img.astype(np.uint8)

            #     cam_param = {}
            #     cam_param['focal'] = (5000, 5000)
            #     cam_param['princpt'] = (192/2, 256/2)
            #     rendered_img = render_mesh(img, mesh_out_cam_pred, smpl_x.face, cam_param)
            #     cv2.imshow("img", rendered_img.astype(np.uint8))
            #     cv2.waitKey(0)
            #     # cv2.imwrite('obj/'+file_name + '.jpg', img)
            #     # save_obj(mesh_gt_cam, smpl_x.face, 'obj/'+file_name + '_gt.obj')
            #     # save_obj(mesh_out_cam, smpl_x.face, 'obj/'+file_name + '.obj')

        return eval_result



    def infer_all(self, model_path):
        outs=[]
        for idx in range(0,len(self.datalist)):
            # for idx in range(0,200):
            data = copy.deepcopy(self.datalist[idx])
            img_path, img_shape, bbox, smplx_param = data['img_path'], data['img_shape'], data['bbox'], data['smplx_param']
            # print("img_path, img_shape, bbox, smplx_param :",img_path, img_shape, bbox, smplx_param )
            cam_param = smplx_param['cam_param']
            img = load_img(img_path)
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, dsplit)

            # anno
            joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig = process_human_model_output(
                smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', None)
            

            ### predict :
            # print("model_path:",model_path)
            print("img_path:",img_path)
            pred_anno= self.infer_one(img_path,model_path)
            joint_img_ori_pred, joint_img_pred, joint_cam_pred, joint_trunc_pred, pose_pred, shape_pred, mesh_cam_orig_pred = process_human_model_output(
                smplx_param['smplx_param'], cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx', pred_anno)
            # outs['smplx_mesh_cam'].append(mesh_cam_orig_pred)
            outs.append({'img': img,'smplx_mesh_cam_target':mesh_cam_orig,'smplx_mesh_cam':mesh_cam_orig_pred})

        return outs

    def infer_one(self,src_path, model_path):
        model = DetBodyHandHead(model_path=model_path)

        person_bbox = []
        result = model(src_path, person_bbox)
        # print("result:",result)
        # print("result:",result[0])
        bbox_body, bbox_lhand, bbox_rhand, bbox_head, [kpt_body, kpt_lhand, kpt_rhand], [kpt_3dbody] = result

        return {'root_pose':kpt_3dbody[0:3] , 'body_pose':kpt_3dbody[-63:]}
    
    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']))
        # f.write(f"{np.mean(eval_result['mpjpe_body'])},{np.mean(eval_result['pa_mpjpe_body'])}")

    def write_eval_result(self, eval_result, subdir):
        with open('../eval_csvs/'+subdir+'_eval.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["sub",subdir])
            writer.writerow(["MPJPE",np.mean(eval_result['mpjpe'])])
            writer.writerow(["PA MPJPE", np.mean(eval_result['pa_mpjpe'])])
            writer.writerow(["MPVPE",np.mean(eval_result['mpvpe'])])
            writer.writerow(["PA MPVPE",np.mean(eval_result['pa_mpvpe'])])


# validate gt labels, visulization
if __name__ == '__main__':
    # load data
    # subdirs=['ConductMusic','Interview','Movie','SignLanguage','TalkShow','Entertainment',\
    #          'LiveVlog','Olympic','Singing','TVShow','Fitness','Magic_show','Online_class',\
    #             'Speech ','VideoConference']

    ## valid subdirs
    subdirs=['Interview','SignLanguage','TalkShow','Entertainment','LiveVlog',\
             'Olympic','TVShow','Fitness','Magic_show','Online_class','VideoConference']

    dsplit = 'train'

    model_path = os.path.join(root_dir, 'runs/pose', '3dat19', 'weights/last.pt')    

    os.makedirs('../eval_csvs/', exist_ok=True)
    
    ### evaluate metrics
    if 0:
        ### single test
        subdir='SignLanguage'
        mpii = MPII(transform=None, data_split=dsplit, subdir=subdir)
        infer_outs=mpii.infer_all(model_path)
        eval_result=mpii.evaluate(infer_outs,0)
        mpii.print_eval_result(eval_result)
        mpii.write_eval_result(eval_result,subdir)

    else:
        ### test all subdirs 
        for subdir in subdirs:
            mpii = MPII(transform=None, data_split=dsplit, subdir=subdir)
            infer_outs=mpii.infer_all(model_path)
            eval_result=mpii.evaluate(infer_outs,0)
            mpii.print_eval_result(eval_result)
            mpii.write_eval_result(eval_result,subdir)