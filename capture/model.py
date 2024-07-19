import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import PositionNet, RotationNet
from nets.loss import CoordLoss, ParamLoss
from utils.human_models import smpl, mano, smpl_x
from utils.transforms import rot6d_to_axis_angle
import math
import copy
import time

body_orig_joint_num = 24
hand_orig_joint_num = 16

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)



class Model(nn.Module):
    def __init__(self, pretrained_weight, resnet_type=50, mode='train', parts='body'):
        super(Model, self).__init__()
        self.parts = parts
        self.mode = mode
        
        # model
        self.backbone = ResNetBackbone(resnet_type, pretrained_weight)
        self.position_net = PositionNet(parts)
        self.rotation_net = RotationNet(parts)

        # init weights
        if mode == 'train':
            self.backbone.init_weights()
            self.position_net.apply(init_weights)
            self.rotation_net.apply(init_weights)
        
        # recon model
        self.smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
        self.smpl_layer = copy.deepcopy(smpl.layer['neutral']).cuda()
        self.mano_layer = copy.deepcopy(mano.layer['right']).cuda()

        # params
        if parts == 'body':
            self.bbox_3d_size = 2
            self.camera_3d_size = 2.5
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
        if parts == 'hand':
            self.bbox_3d_size = 0.3
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 256)
            self.output_hm_shape = (8, 8, 8)
        self.focal = (5000, 5000) # virtual focal lengths
        self.princpt = (self.input_img_shape[1]/2, self.input_img_shape[0]/2)

        # loss
        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(self.focal[0]*self.focal[1]*self.camera_3d_size*self.camera_3d_size/(self.input_img_shape[0]*self.input_img_shape[1]))]).cuda().view(-1)
        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def get_coord_old(self, params):
        batch_size = params['root_pose'].shape[0]

        if self.parts == 'body':
            output = self.smpl_layer(global_orient=params['root_pose'], body_pose=params['body_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(smpl.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = smpl.root_joint_idx
        elif self.parts == 'hand':
            output = self.mano_layer(global_orient=params['root_pose'], hand_pose=params['hand_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = mano.root_joint_idx

        # project 3D coordinates to 2D space
        cam_trans = params['cam_trans']
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.focal[0] + self.princpt[0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.focal[1] + self.princpt[1]

        x = x / self.input_img_shape[1] * self.output_hm_shape[2]
        y = y / self.input_img_shape[0] * self.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]
        return joint_proj, joint_cam, mesh_cam

    def get_coord(self, params):
        batch_size = params['root_pose'].shape[0]
        zero_pose = torch.zeros((batch_size, 3)).float().cuda()
        expr = torch.zeros((batch_size, 10)).float().cuda()
        hand_pose = torch.zeros((batch_size, 45)).float().cuda()

        if self.parts == 'body':

            # output = self.smpl_layer(global_orient=params['root_pose'], body_pose=params['body_pose'], betas=params['shape'])
            output = self.smplx_layer(betas=params['shape'], body_pose=params['body_pose'], global_orient=params['root_pose'],
                                          transl=zero_pose, left_hand_pose=hand_pose,
                                          right_hand_pose=hand_pose, jaw_pose=zero_pose,
                                          leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)

            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = output.joints[:, smpl_x.joint_idx, :]  # (b,137,3)
            joint_cam = joint_cam[:, 0:25, :]  # (25,3)
            root_joint_idx = smpl_x.root_joint_idx
        elif self.parts == 'hand':
            output = self.mano_layer(global_orient=params['root_pose'], hand_pose=params['hand_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = mano.root_joint_idx

        # project 3D coordinates to 2D space
        cam_trans = params['cam_trans']
        x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.focal[0] + self.princpt[0]
        y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.focal[1] + self.princpt[1]

        x = x / self.input_img_shape[1] * self.output_hm_shape[2]
        y = y / self.input_img_shape[0] * self.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]
        return joint_proj, joint_cam, mesh_cam


    def forward(self, inputs, targets, meta_info):
        # t = time.time()
        img_feat = self.backbone(inputs['img'])
        batch_size = img_feat.shape[0]
        joint_img = self.position_net(img_feat)
        root_pose_6d, pose_param_6d, shape_param, cam_param = self.rotation_net(img_feat, joint_img)
        # print("network forward: ", time.time() - t)
        root_pose = rot6d_to_axis_angle(root_pose_6d)
        body_pose = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(batch_size,-1)
        cam_trans = self.get_camera_trans(cam_param)
        smplx_pose_6d = torch.cat((root_pose_6d, pose_param_6d),1)
        if self.parts == 'body':
            # pose_param = torch.cat((pose_param, torch.zeros((batch_size,2*3)).cuda().float()),1)
            # t = time.time()
            joint_proj, joint_cam, mesh_cam = self.get_coord({'root_pose': root_pose, 'body_pose': body_pose, 'shape': shape_param, 'cam_trans': cam_trans})
            # print("self.get_coord: ", time.time() - t)
            smpl_pose = torch.cat((root_pose, body_pose),1)
        
        elif self.parts == 'hand':
            pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(-1,(hand_orig_joint_num-1)*3)
            joint_proj, joint_cam, mesh_cam = self.get_coord({'root_pose': root_pose, 'hand_pose': pose_param, 'shape': shape_param, 'cam_trans': cam_trans})
            mano_hand_pose = pose_param.view(-1,(mano.orig_joint_num-1)*3)
            mano_pose = torch.cat((root_pose, mano_hand_pose),1)


        if self.mode == 'train':
            # loss functions
            loss = {}
            if self.parts == 'body':
                # 2d joint prediction by positionnet  == gt 3d joint projection
                loss['smpl_joint_img'] = self.coord_loss(joint_img, targets['smplx_joint_img'], meta_info['smplx_joint_trunc'])
                # loss['smpl_pose'] = self.param_loss(smpl_pose, targets['smplx_pose'], meta_info['smplx_pose_valid']) # computing loss with rotation matrix instead of axis-angle can avoid ambiguity of axis-angle. current: compute loss with axis-angle. should be fixed.
                loss['smpl_pose_6d'] = 2.0 * self.param_loss(smplx_pose_6d, targets['smplx_pose_6d'], meta_info['smplx_pose_6d_valid'])
                loss['smpl_shape'] = 1.0 * self.param_loss(shape_param, targets['smplx_shape'], meta_info['smplx_shape_valid'][:,None])
                # root-relative 3d smplx joints prediction = gt 3d smplx ones
                loss['smpl_joint_cam'] = 4.0 * self.coord_loss(joint_cam, targets['smplx_joint_cam'], meta_info['smplx_joint_valid'])
                # root-relative 3d smplx joints projection = gt 2d labeled ones
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['smplx_joint_img'][:,:,:2], meta_info['smplx_joint_trunc'])

                # # root-relative 3d smplx joints prediction = gt 3d labeled ones
                # loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])

                # # 2d joint prediction by positionnet  == gt 2d joint labels
                # loss['joint_img'] = self.coord_loss(joint_img, smpl.reduce_joint_set(targets['joint_img']), smpl.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D'])


            elif self.parts == 'hand':
                loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
                loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'], meta_info['mano_joint_trunc'])
                loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_pose_valid']) # computing loss with rotation matrix instead of axis-angle can avoid ambiguity of axis-angle. current: compute loss with axis-angle. should be fixed.
                loss['mano_shape'] = self.param_loss(shape_param, targets['mano_shape'], meta_info['mano_shape_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'], meta_info['mano_joint_valid'])
            return loss
        else:
            # test output
            out = {'cam_trans': cam_trans} 
            if self.parts == 'body':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img
                out['smplx_joint_cam'] = joint_cam
                out['smplx_mesh_cam'] = mesh_cam
                out['smplx_joint_proj'] = joint_proj
                out['smplx_pose'] = smpl_pose
                out['smplx_shape'] = shape_param
                out['smplx_joint_cam_target'] = targets['smplx_joint_cam']
                out['smplx_mesh_cam_target'] = targets['smplx_mesh_cam']
                # if 'bb2img_trans' in meta_info:
                #     out['bb2img_trans'] = meta_info['bb2img_trans']
            elif self.parts == 'hand':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img 
                out['mano_mesh_cam'] = mesh_cam
                out['mano_pose'] = mano_pose
                out['mano_shape'] = shape_param
                if 'mano_mesh_cam' in targets:
                    out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
                if 'joint_img' in targets:
                    out['joint_img_target'] = targets['joint_img']
                if 'joint_valid' in meta_info:
                    out['joint_valid'] = meta_info['joint_valid']
                if 'bb2img_trans' in meta_info:
                    out['bb2img_trans'] = meta_info['bb2img_trans']
            return out
        

class InferModel(nn.Module):
    # convert to onnx/trt
    def __init__(self, pretrained_weight, resnet_type=50, parts='body'):
        super(InferModel, self).__init__()
        self.parts = parts

        self.backbone = ResNetBackbone(resnet_type, pretrained_weight)
        self.position_net = PositionNet(parts)
        self.rotation_net = RotationNet(parts)

    def forward(self, inputs):

        img_feat = self.backbone(inputs['img'])
        joint_img = self.position_net(img_feat)
        root_pose_6d, pose_param_6d, _, _ = self.rotation_net(img_feat, joint_img)
        return root_pose_6d, pose_param_6d
    

def combine_pose(root_pose, pose, parts='body'):
    # post-process InferModel results
    batch_size = root_pose.shape[0]
    root_pose = rot6d_to_axis_angle(root_pose)
    if parts == 'body':
        pose = rot6d_to_axis_angle(pose.view(-1,6)).reshape(1,-1)
        pose = torch.cat((pose, torch.zeros((batch_size, 2*3)).cuda().float()),1)
        pose = pose.view(-1,(body_orig_joint_num-1)*3)
    elif parts == 'hand':
        pose = rot6d_to_axis_angle(pose.view(-1,6)).reshape(-1,(hand_orig_joint_num-1)*3)

    combine_pose = torch.cat((root_pose, pose), 1)
    return combine_pose