import numpy as np
import cv2
import random
# from config import cfg
import math
from utils.human_models import smpl, smpl_x, mano, flame
from utils.transforms import cam2pixel, transform_joint_to_other_db
from plyfile import PlyData, PlyElement
import torch


bbox_3d_size = 2
camera_3d_size = 2.5
input_img_shape = (256, 192)
output_hm_shape = (8, 8, 6)

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid, extend_ratio=1.2):

    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, img_width, img_height, do_sanitize=True):
    if do_sanitize:
        # sanitize bboxes
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
        if w*h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1])
        else:
            return None

   # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = input_img_shape[1]/input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    
    bbox = bbox.astype(np.float32)
    return bbox

def get_aug_config():
    # scale_factor = 0.25
    # rot_factor = 30
    # color_factor = 0.2
    scale_factor = 0.2
    rot_factor = 20.
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip

def augmentation(img, bbox, data_split, enforce_flip=None):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    if enforce_flip is None:
        pass
    elif enforce_flip is True:
        do_flip = True
    elif enforce_flip is False:
        do_flip = False
    
    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, input_img_shape)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, do_flip

def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def process_db_coord(joint_img, joint_cam, joint_valid, do_flip, img_shape, flip_pairs, img2bb_trans, rot, src_joints_name, target_joints_name):
    joint_img, joint_cam, joint_valid = joint_img.copy(), joint_cam.copy(), joint_valid.copy()

    # flip augmentation
    if do_flip:
        joint_cam[:,0] = -joint_cam[:,0]
        joint_img[:,0] = img_shape[1] - 1 - joint_img[:,0]
        for pair in flip_pairs:
            joint_img[pair[0],:], joint_img[pair[1],:] = joint_img[pair[1],:].copy(), joint_img[pair[0],:].copy()
            joint_cam[pair[0],:], joint_cam[pair[1],:] = joint_cam[pair[1],:].copy(), joint_cam[pair[0],:].copy()
            joint_valid[pair[0],:], joint_valid[pair[1],:] = joint_valid[pair[1],:].copy(), joint_valid[pair[0],:].copy()
    
    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
    [0, 0, 1]], dtype=np.float32)
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1,0)).transpose(1,0)

    # affine transformation and root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
    joint_img[:,:2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1,0)).transpose(1,0)
    joint_img[:,0] = joint_img[:,0] / input_img_shape[1] * output_hm_shape[2]
    joint_img[:,1] = joint_img[:,1] / input_img_shape[0] * output_hm_shape[1]
    joint_img[:,2] = (joint_img[:,2] / (bbox_3d_size / 2) + 1)/2. * output_hm_shape[0]
    
    # check truncation
    joint_trunc = joint_valid * ((joint_img[:,0] >= 0) * (joint_img[:,0] < output_hm_shape[2]) * \
                (joint_img[:,1] >= 0) * (joint_img[:,1] < output_hm_shape[1]) * \
                (joint_img[:,2] >= 0) * (joint_img[:,2] < output_hm_shape[0])).reshape(-1,1).astype(np.float32)

    # transform joints to target db joints
    joint_img = transform_joint_to_other_db(joint_img, src_joints_name, target_joints_name)
    joint_cam = transform_joint_to_other_db(joint_cam, src_joints_name, target_joints_name)
    joint_valid = transform_joint_to_other_db(joint_valid, src_joints_name, target_joints_name)
    joint_trunc = transform_joint_to_other_db(joint_trunc, src_joints_name, target_joints_name)
    return joint_img, joint_cam, joint_valid, joint_trunc


#joint_img_ori: (22,2) in x,y
#joint_img: (22,3) in x,y,d
#joint_cam:(22,3) in root-relative X,Y,Z in cam-sys
#joint_trunc: (22,1)
#pose: (66,)
#shape: (10,)
#mesh_cam_orig: (10475,3)
def process_human_model_output_old(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, human_model_type):
    if human_model_type == 'smplx':
        human_model = smpl_x
        # rotation_valid = np.ones((smpl_x.orig_joint_num), dtype=np.float32)
        # coord_valid = np.ones((smpl_x.joint_num), dtype=np.float32)
        rotation_valid = np.ones(smpl_x.orig_body_joint_num, dtype=np.float32)
        coord_valid = np.ones((smpl_x.pos_joint_num), dtype=np.float32)

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
        human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param and human_model_param['lhand_valid']:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['lhand']] = 0
            # coord_valid[smpl_x.joint_part['lhand']] = 0
        if 'rhand_pose' in human_model_param and human_model_param['rhand_valid']:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['rhand']] = 0
            # coord_valid[smpl_x.joint_part['rhand']] = 0
        if 'jaw_pose' in human_model_param and 'expr' in human_model_param and human_model_param['face_valid']:
            jaw_pose = human_model_param['jaw_pose']
            expr = human_model_param['expr']
            expr_valid = True
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['face']] = 0
            # coord_valid[smpl_x.joint_part['face']] = 0
            expr_valid = False
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).view(1, 3)

        # get mesh and joint coordinates
        zero_pose = torch.zeros((1, 3)).float()  # eye poses
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
        mesh_cam = output.vertices[0].numpy()
        # joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :] #(137,3)
        # joint_cam = joint_cam[0:25,:] #(25,3)
        joint_cam = np.dot(smpl_x.J_regressor, mesh_cam) # (55,3)
        joint_cam = joint_cam[0:22, :]

        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                      dtype=np.float32).reshape(1, 3)
            root_cam = joint_cam[smpl_x.root_joint_idx, None, :]
            joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0) + t
            mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0) + t

        # concat root, body, two hands, and jaw pose
        # pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose))
        pose = torch.cat((root_pose, body_pose))

        # joint coordinates
        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_img_ori = joint_img.copy()[:, 0:2] #(22,2)
        joint_cam = joint_cam - joint_cam[smpl_x.orig_root_joint_idx, None, :]  # root-relative
        # joint_cam[smpl_x.joint_part['lhand'], :] = joint_cam[smpl_x.joint_part['lhand'], :] - joint_cam[
        #                                                                                       smpl_x.lwrist_idx, None,
        #                                                                                       :]  # left hand root-relative
        # joint_cam[smpl_x.joint_part['rhand'], :] = joint_cam[smpl_x.joint_part['rhand'], :] - joint_cam[
        #                                                                                       smpl_x.rwrist_idx, None,
        #                                                                                       :]  # right hand root-relative
        # joint_cam[smpl_x.joint_part['face'], :] = joint_cam[smpl_x.joint_part['face'], :] - joint_cam[smpl_x.neck_idx,
        #                                                                                     None,
        #                                                                                     :]  # face root-relative
        joint_img[smpl_x.orig_joint_part['body'], 2] = (joint_cam[smpl_x.orig_joint_part['body'], 2].copy() / (
                    bbox_3d_size / 2) + 1) / 2. * output_hm_shape[0]  # body depth discretize
        # joint_img[smpl_x.joint_part['lhand'], 2] = (joint_cam[smpl_x.joint_part['lhand'], 2].copy() / (
        #             cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # left hand depth discretize
        # joint_img[smpl_x.joint_part['rhand'], 2] = (joint_cam[smpl_x.joint_part['rhand'], 2].copy() / (
        #             cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # right hand depth discretize
        # joint_img[smpl_x.joint_part['face'], 2] = (joint_cam[smpl_x.joint_part['face'], 2].copy() / (
        #             cfg.face_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # face depth discretize



    mesh_cam_orig = mesh_cam.copy()  # back-up the original one

    ## so far, data augmentations are not applied yet
    ## now, apply data augmentations

    # image projection
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in human_model.orig_flip_body_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :].copy(), joint_cam[pair[0], :].copy()
            joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :].copy(), joint_img[pair[0], :].copy()
            # if human_model_type == 'smplx':
            #     coord_valid[pair[0]], coord_valid[pair[1]] = coord_valid[pair[1]].copy(), coord_valid[pair[0]].copy()

    # x,y affine transform, root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, 0:1])), 1)
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    joint_img[:, 0] = joint_img[:, 0] / input_img_shape[1] * output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / input_img_shape[0] * output_hm_shape[1]

    # check truncation
    joint_trunc = ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < output_hm_shape[2]) * \
                   (joint_img[:, 1] >= 0) * (joint_img[:, 1] < output_hm_shape[1]) * \
                   (joint_img[:, 2] >= 0) * (joint_img[:, 2] < output_hm_shape[0])).reshape(-1, 1).astype(
        np.float32)

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                            [0, 0, 1]], dtype=np.float32)
    # coordinate
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)
    # parameters
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_body_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
            if human_model_type == 'smplx':
                rotation_valid[pair[0]], rotation_valid[pair[1]] = rotation_valid[pair[1]].copy(), rotation_valid[
                    pair[0]].copy()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle

    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[human_model.orig_root_joint_idx, :]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
    pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)

    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(-1)

    # return results
    if human_model_type == 'smplx':
        pose = pose.reshape(-1)
        expr = expr.numpy().reshape(-1)
        return joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig

def process_human_model_output_new(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, human_model_type):
    if human_model_type == 'smplx':
        human_model = smpl_x

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
        human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * 15), dtype=np.float32)
        if 'rhand_pose' in human_model_param:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * 15), dtype=np.float32)
        if 'jaw_pose' in human_model_param:
            jaw_pose = human_model_param['jaw_pose']
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
        if 'leye_pose' in human_model_param:
            leye_pose = human_model_param['leye_pose']
        else:
            leye_pose = np.zeros((3), dtype=np.float32)
        if 'reye_pose' in human_model_param:
            reye_pose = human_model_param['reye_pose']
        else:
            reye_pose = np.zeros((3), dtype=np.float32)
        if 'expression' in human_model_param:
            expression = human_model_param['expression']
        else:
            expression = np.zeros((10), dtype=np.float32)
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        leye_pose = torch.FloatTensor(leye_pose).view(-1, 3)  # (1,3)
        reye_pose = torch.FloatTensor(reye_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expression).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector
        pose = torch.cat((root_pose, body_pose))

        # get mesh and joint coordinates
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=leye_pose, reye_pose=reye_pose, expression=expr)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :] #(137,3)
        joint_cam = joint_cam[0:25,:] #(25,3)
        # joint_cam = np.dot(smpl_x.J_regressor, mesh_cam) # (55,3)
        # joint_cam = joint_cam[0:22, :]

        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_img_ori = joint_img.copy()[:, 0:2] #(25,2)
        joint_cam = joint_cam - joint_cam[smpl_x.orig_root_joint_idx, None, :]  # root-relative
        joint_img[smpl_x.orig_joint_part['body'], 2] = (joint_cam[smpl_x.orig_joint_part['body'], 2].copy() / (
                    bbox_3d_size / 2) + 1) / 2. * output_hm_shape[0]  # body depth discretize



    mesh_cam_orig = mesh_cam.copy()  # back-up the original one

    # image projection
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in human_model.flip_body_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :].copy(), joint_cam[pair[0], :].copy()
            joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :].copy(), joint_img[pair[0], :].copy()

    # x,y affine transform, root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, 0:1])), 1)
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    joint_img_crop = joint_img[:, :2].copy()
    joint_img[:, 0] = joint_img[:, 0] / input_img_shape[1] * output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / input_img_shape[0] * output_hm_shape[1]

    # check truncation
    joint_trunc = ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < output_hm_shape[2]) * \
                   (joint_img[:, 1] >= 0) * (joint_img[:, 1] < output_hm_shape[1]) * \
                   (joint_img[:, 2] >= 0) * (joint_img[:, 2] < output_hm_shape[0])).reshape(-1, 1).astype(
        np.float32)

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                            [0, 0, 1]], dtype=np.float32)
    # coordinate
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)

    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_body_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle

    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[human_model.orig_root_joint_idx, :]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
    pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)

    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(-1)

    # return results
    if human_model_type == 'smplx':
        pose = pose.reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig, joint_img_ori, joint_img_crop


def process_human_model_output_new_debug(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, human_model_type):
    if human_model_type == 'smplx':
        human_model = smpl_x

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
        human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * 15), dtype=np.float32)
        if 'rhand_pose' in human_model_param:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * 15), dtype=np.float32)
        if 'jaw_pose' in human_model_param:
            jaw_pose = human_model_param['jaw_pose']
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
        if 'leye_pose' in human_model_param:
            leye_pose = human_model_param['leye_pose']
        else:
            leye_pose = np.zeros((3), dtype=np.float32)
        if 'reye_pose' in human_model_param:
            reye_pose = human_model_param['reye_pose']
        else:
            reye_pose = np.zeros((3), dtype=np.float32)
        if 'expression' in human_model_param:
            expression = human_model_param['expression']
        else:
            expression = np.zeros((10), dtype=np.float32)
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        leye_pose = torch.FloatTensor(leye_pose).view(-1, 3)  # (1,3)
        reye_pose = torch.FloatTensor(reye_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expression).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector
        pose = torch.cat((root_pose, body_pose))

        # get mesh and joint coordinates
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=leye_pose, reye_pose=reye_pose, expression=expr)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :] #(137,3)
        joint_cam = joint_cam[0:25,:] #(25,3)
        # joint_cam = np.dot(smpl_x.J_regressor, mesh_cam) # (55,3)
        # joint_cam = joint_cam[0:22, :]

        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_img_ori = joint_img.copy()[:, 0:2] #(25,2)
        joint_cam = joint_cam - joint_cam[smpl_x.orig_root_joint_idx, None, :]  # root-relative
        joint_img[smpl_x.orig_joint_part['body'], 2] = (joint_cam[smpl_x.orig_joint_part['body'], 2].copy() / (
                    bbox_3d_size / 2) + 1) / 2. * output_hm_shape[0]  # body depth discretize



    mesh_cam_orig = mesh_cam.copy()  # back-up the original one

    # image projection
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in human_model.flip_body_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :].copy(), joint_cam[pair[0], :].copy()
            joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :].copy(), joint_img[pair[0], :].copy()

    # x,y affine transform, root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, 0:1])), 1)
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    joint_img_crop = joint_img[:, :2].copy()
    joint_img[:, 0] = joint_img[:, 0] / input_img_shape[1] * output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / input_img_shape[0] * output_hm_shape[1]

    # check truncation
    joint_trunc = ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < output_hm_shape[2]) * \
                   (joint_img[:, 1] >= 0) * (joint_img[:, 1] < output_hm_shape[1]) * \
                   (joint_img[:, 2] >= 0) * (joint_img[:, 2] < output_hm_shape[0])).reshape(-1, 1).astype(
        np.float32)

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                            [0, 0, 1]], dtype=np.float32)
    # coordinate
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)

    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_body_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle

    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[human_model.orig_root_joint_idx, :]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
    pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)

    # # change to mean shape if beta is too far from it
    # shape[(shape.abs() > 3).any(dim=1)] = 0.
    # shape = shape.numpy().reshape(-1)

    root_pose = torch.FloatTensor(pose[human_model.orig_root_joint_idx]).view(1, 3)  # (1,3)
    body_pose = torch.FloatTensor(pose[1:, :]).view(-1, 3)  # (21,3)
    # get mesh and joint coordinates
    with torch.no_grad():
        output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                      transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                      right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                      leye_pose=leye_pose, reye_pose=reye_pose, expression=expr)
    mesh_cam_aug = output.vertices[0].numpy()
    mesh_cam_aug = mesh_cam_aug + np.array((0,0, 60))

    # return results
    if human_model_type == 'smplx':
        pose = pose.reshape(-1)
        return joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig, mesh_cam_aug, joint_img_ori, joint_img_crop

def process_human_model_output(human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, human_model_type):
    if human_model_type == 'smplx':
        human_model = smpl_x
        # rotation_valid = np.ones((smpl_x.orig_joint_num), dtype=np.float32)
        # coord_valid = np.ones((smpl_x.joint_num), dtype=np.float32)
        # rotation_valid = np.ones(smpl_x.orig_body_joint_num, dtype=np.float32)
        # coord_valid = np.ones((smpl_x.pos_joint_num), dtype=np.float32)

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
        human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param and human_model_param['lhand_valid']:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['lhand']] = 0
            # coord_valid[smpl_x.joint_part['lhand']] = 0
        if 'rhand_pose' in human_model_param and human_model_param['rhand_valid']:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['rhand']] = 0
            # coord_valid[smpl_x.joint_part['rhand']] = 0
        if 'jaw_pose' in human_model_param and 'expr' in human_model_param and human_model_param['face_valid']:
            jaw_pose = human_model_param['jaw_pose']
            expr = human_model_param['expr']
            expr_valid = True
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            # rotation_valid[smpl_x.orig_joint_part['face']] = 0
            # coord_valid[smpl_x.joint_part['face']] = 0
            expr_valid = False
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).view(1, 3)

        # get mesh and joint coordinates
        zero_pose = torch.zeros((1, 3)).float()  # eye poses
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
        mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :] #(137,3)
        joint_cam = joint_cam[0:25,:] #(25,3)
        # joint_cam = np.dot(smpl_x.J_regressor, mesh_cam) # (55,3)
        # joint_cam = joint_cam[0:22, :]

        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                      dtype=np.float32).reshape(1, 3)
            root_cam = joint_cam[smpl_x.root_joint_idx, None, :]
            joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0) + t
            mesh_cam = mesh_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0) + t

            # root_cam_25 = joint_cam_25[smpl_x.root_joint_idx, None, :]
            # joint_cam_25 = joint_cam_25 - root_cam_25 + np.dot(R, root_cam_25.transpose(1, 0)).transpose(1, 0) + t

        # concat root, body, two hands, and jaw pose
        # pose = torch.cat((root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose))
        pose = torch.cat((root_pose, body_pose))

        # joint coordinates
        # joint_img_25 = cam2pixel(joint_cam_25, cam_param['focal'], cam_param['princpt'])
        # joint_img_ori_25 = joint_img_25.copy()[:, 0:2]  # (22,2)
        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_img_ori = joint_img.copy()[:, 0:2] #(25,2)
        joint_cam = joint_cam - joint_cam[smpl_x.orig_root_joint_idx, None, :]  # root-relative
        # joint_cam[smpl_x.joint_part['lhand'], :] = joint_cam[smpl_x.joint_part['lhand'], :] - joint_cam[
        #                                                                                       smpl_x.lwrist_idx, None,
        #                                                                                       :]  # left hand root-relative
        # joint_cam[smpl_x.joint_part['rhand'], :] = joint_cam[smpl_x.joint_part['rhand'], :] - joint_cam[
        #                                                                                       smpl_x.rwrist_idx, None,
        #                                                                                       :]  # right hand root-relative
        # joint_cam[smpl_x.joint_part['face'], :] = joint_cam[smpl_x.joint_part['face'], :] - joint_cam[smpl_x.neck_idx,
        #                                                                                     None,
        #                                                                                     :]  # face root-relative
        joint_img[smpl_x.orig_joint_part['body'], 2] = (joint_cam[smpl_x.orig_joint_part['body'], 2].copy() / (
                    bbox_3d_size / 2) + 1) / 2. * output_hm_shape[0]  # body depth discretize
        # joint_img[smpl_x.joint_part['lhand'], 2] = (joint_cam[smpl_x.joint_part['lhand'], 2].copy() / (
        #             cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # left hand depth discretize
        # joint_img[smpl_x.joint_part['rhand'], 2] = (joint_cam[smpl_x.joint_part['rhand'], 2].copy() / (
        #             cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # right hand depth discretize
        # joint_img[smpl_x.joint_part['face'], 2] = (joint_cam[smpl_x.joint_part['face'], 2].copy() / (
        #             cfg.face_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # face depth discretize



    mesh_cam_orig = mesh_cam.copy()  # back-up the original one

    ## so far, data augmentations are not applied yet
    ## now, apply data augmentations

    # image projection
    if do_flip:
        joint_cam[:, 0] = -joint_cam[:, 0]
        joint_img[:, 0] = img_shape[1] - 1 - joint_img[:, 0]
        for pair in human_model.flip_body_pairs:
            joint_cam[pair[0], :], joint_cam[pair[1], :] = joint_cam[pair[1], :].copy(), joint_cam[pair[0], :].copy()
            joint_img[pair[0], :], joint_img[pair[1], :] = joint_img[pair[1], :].copy(), joint_img[pair[0], :].copy()
            # if human_model_type == 'smplx':
            #     coord_valid[pair[0]], coord_valid[pair[1]] = coord_valid[pair[1]].copy(), coord_valid[pair[0]].copy()

    # x,y affine transform, root-relative depth
    joint_img_xy1 = np.concatenate((joint_img[:, :2], np.ones_like(joint_img[:, 0:1])), 1)
    joint_img[:, :2] = np.dot(img2bb_trans, joint_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    joint_img[:, 0] = joint_img[:, 0] / input_img_shape[1] * output_hm_shape[2]
    joint_img[:, 1] = joint_img[:, 1] / input_img_shape[0] * output_hm_shape[1]

    # check truncation
    joint_trunc = ((joint_img[:, 0] >= 0) * (joint_img[:, 0] < output_hm_shape[2]) * \
                   (joint_img[:, 1] >= 0) * (joint_img[:, 1] < output_hm_shape[1]) * \
                   (joint_img[:, 2] >= 0) * (joint_img[:, 2] < output_hm_shape[0])).reshape(-1, 1).astype(
        np.float32)

    # 3D data rotation augmentation
    rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                            [0, 0, 1]], dtype=np.float32)
    # coordinate
    joint_cam = np.dot(rot_aug_mat, joint_cam.transpose(1, 0)).transpose(1, 0)
    # parameters
    # flip pose parameter (axis-angle)
    if do_flip:
        for pair in human_model.orig_flip_body_pairs:
            pose[pair[0], :], pose[pair[1], :] = pose[pair[1], :].clone(), pose[pair[0], :].clone()
            # if human_model_type == 'smplx':
            #     rotation_valid[pair[0]], rotation_valid[pair[1]] = rotation_valid[pair[1]].copy(), rotation_valid[
            #         pair[0]].copy()
        pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle

    # rotate root pose
    pose = pose.numpy()
    root_pose = pose[human_model.orig_root_joint_idx, :]
    root_pose, _ = cv2.Rodrigues(root_pose)
    root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
    pose[human_model.orig_root_joint_idx] = root_pose.reshape(3)

    # change to mean shape if beta is too far from it
    shape[(shape.abs() > 3).any(dim=1)] = 0.
    shape = shape.numpy().reshape(-1)

    # return results
    if human_model_type == 'smplx':
        pose = pose.reshape(-1)
        expr = expr.numpy().reshape(-1)
        return joint_img_ori, joint_img, joint_cam, joint_trunc, pose, shape, mesh_cam_orig


def load_obj(file_name):
    v = []
    obj_file = open(file_name)
    for line in obj_file:
        words = line.split(' ')
        if words[0] == 'v':
            x,y,z = float(words[1]), float(words[2]), float(words[3])
            v.append(np.array([x,y,z]))
    return np.stack(v)



def resize_bbox(bbox, scale=1.2):
    if isinstance(bbox, list):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    else:
        x1, y1, x2, y2 = bbox
    x_center = (x1+x2)/2.0
    y_center = (y1+y2)/2.0
    x_size, y_size = x2-x1, y2-y1
    x1_resize = x_center-x_size/2.0*scale
    x2_resize = x_center+x_size/2.0*scale
    y1_resize = y_center - y_size / 2.0 * scale
    y2_resize = y_center + y_size / 2.0 * scale
    bbox[0], bbox[1], bbox[2], bbox[3] = x1_resize, y1_resize, x2_resize, y2_resize
    return bbox