import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from .layer import make_conv_layers, make_linear_layers
from utils.transforms import sample_joint_features, soft_argmax_3d

body_output_hm_shape = (8, 8, 6)
hand_output_hm_shape = (8, 8, 8)

body_joint_num = 25
hand_joint_num = 21
body_orig_joint_num = 24
body_shape_param_dim = 10
hand_orig_joint_num = 16
hand_shape_param_dim = 10

class PositionNet(nn.Module):
    def __init__(self, parts='body'):
        super(PositionNet, self).__init__()
        self.parts = parts
        if parts == 'body':
            self.joint_num = body_joint_num
            self.output_hm_shape = body_output_hm_shape
        elif parts == 'hand':
            self.joint_num = hand_joint_num
            self.output_hm_shape = hand_output_hm_shape
        self.conv = make_conv_layers([2048,self.joint_num*self.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,self.output_hm_shape[0],self.output_hm_shape[1],self.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord

class RotationNet(nn.Module):
    def __init__(self, parts='body'):
        super(RotationNet, self).__init__()
        self.parts = parts
        if parts == 'body':
            self.joint_num = body_joint_num
        elif parts == 'hand':
            self.joint_num = hand_joint_num
       
        # output layers
        if parts == 'body':
            self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(512+3), (body_orig_joint_num-3)*6], relu_final=False) # without root and two hands
            self.shape_out = make_linear_layers([2048,body_shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([2048,3], relu_final=False)
        elif parts == 'hand':
            self.conv = make_conv_layers([2048,512], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(512+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(512+3), (hand_orig_joint_num-1)*6], relu_final=False) # without root joint
            self.shape_out = make_linear_layers([2048,hand_shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([2048,3], relu_final=False)

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        img_feat = self.conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)

        root_pose = self.root_pose_out(feat.view(batch_size,-1))
        pose_param = self.pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param

