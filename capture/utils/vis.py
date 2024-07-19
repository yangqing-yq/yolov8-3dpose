import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
# from config import cfg
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh


def vis_kp2d(image, kp2d):
    img2 = image.copy()
    for i in range(kp2d.shape[0]):  # draw keypoints
    # for i in range(22):  # draw keypoints
        temp_x = kp2d[i][0]
        temp_y = kp2d[i][1]
        cv2.circle(img2, (int(temp_x), int(temp_y)), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    return img2

def vis_kp2d_bbox(image, kp2d, box):
    img2 = image.copy()
    for i in range(kp2d.shape[0]):  # draw keypoints
    # for i in range(22):  # draw keypoints
        temp_x = kp2d[i][0]
        temp_y = kp2d[i][1]
        cv2.circle(img2, (int(temp_x), int(temp_y)), 4, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    pt1_x = int(box[0])
    pt1_y = int(box[1])
    pt2_x = int(box[0] + box[2])
    pt2_y = int(box[1] + box[3])
    pt1_x = np.clip(pt1_x, 0, image.shape[1])
    pt1_y = np.clip(pt1_y, 0, image.shape[0])
    pt2_x = np.clip(pt2_x, 0, image.shape[1])
    pt2_y = np.clip(pt2_y, 0, image.shape[0])
    cv2.rectangle(img2, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), (0,255,0), 2)
    return img2

def vis_bbox(image, box):
    img2 = image.copy()
    pt1_x = int(box[0])
    pt1_y = int(box[1])
    pt2_x = int(box[0]+box[2])
    pt2_y = int(box[1]+box[3])
    pt1_x = np.clip(pt1_x, 0, image.shape[1])
    pt1_y = np.clip(pt1_y, 0, image.shape[0])
    pt2_x = np.clip(pt2_x, 0, image.shape[1])
    pt2_y = np.clip(pt2_y, 0, image.shape[0])
    cv2.rectangle(img2, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), (0,255,0), 2)
    return img2

def vis_kp2d_2(image, kp2d, kp2d_2):
    img2 = image.copy()
    # for i in range(kp2d.shape[0]):  # draw keypoints
    for i in range(22):  # draw keypoints
        temp_x = kp2d[i][0]
        temp_y = kp2d[i][1]
        cv2.circle(img2, (int(temp_x), int(temp_y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    for i in range(25):  # draw keypoints
        temp_x = kp2d_2[i][0]
        temp_y = kp2d_2[i][1]
        cv2.circle(img2, (int(temp_x), int(temp_y)), 8, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    return img2

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

# def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
#     cmap = plt.get_cmap('rainbow')
#     colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
#     colors = [np.array((c[2], c[1], c[0])) for c in colors]

#     for l in range(len(kps_lines)):
#         i1 = kps_lines[l][0]
#         i2 = kps_lines[l][1]
#         x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
#         y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
#         z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

#         if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
#             ax.plot(x, z, -y, c=colors[l], linewidth=2)
#         if kpt_3d_vis[i1,0] > 0:
#             ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
#         if kpt_3d_vis[i2,0] > 0:
#             ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

#     x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
#     y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
#     z_r = np.array([0, 1], dtype=np.float32)
    
#     if filename is None:
#         ax.set_title('3D vis')
#     else:
#         ax.set_title(filename)

#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Z Label')
#     ax.set_zlabel('Y Label')
#     ax.legend()

#     plt.show()
#     cv2.waitKey(0)

def render_mesh(img, mesh, face, cam_param):
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

