'''
生成标签数据
类别为4个类：全身、头、左手、右手
关键点取全身17点，左手10点，右手10点
'''


import numpy as np
from pathlib import Path
import posixpath
import shutil
import json
from collections import defaultdict
from collections import OrderedDict
import cv2
from tqdm import tqdm
import os
import time
from multiprocessing import Pool
import codecs

def make_3D_ann(anno_3d_info, type):
    # import pdb; pdb.set_trace()
    # search smplx anno via id
    ann_3D = []

    if type == 'body':
        # zeros_list = ([0] * 3) * 15 * 2
        # ann_3D = anno_3d_info['root_pose'] + anno_3d_info['body_pose'] + zeros_list
        ann_3D = anno_3d_info['root_pose'] + anno_3d_info['body_pose']
    elif type == 'lhand':
        # zeros_list1 = ([0] * 3) * 22
        # zeros_list2 = ([0] * 3) * + 15
        # ann_3D = zeros_list1 + anno_3d_info['lhand_pose'] + zeros_list2
        ann_3D = ([0] * 3) * 22
    elif type == 'rhand':
        # zeros_list1 = ([0] * 3) * 22
        # zeros_list2 = ([0] * 3) * + 15
        # ann_3D = zeros_list1 + zeros_list2 + anno_3d_info['rhand_pose']
        ann_3D = ([0] * 3) * 22
    elif type == 'head':
        # zeros_list = ([0] * 3) * (22 + 15 * 2)
        zeros_list = ([0] * 3) * 22
        return zeros_list
    else:
        print("invalid input")
        # zeros_list = ([0] * 3) * (22 + 15 * 2)
        zeros_list = ([0] * 3) * 22
        return zeros_list

    return ann_3D

def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    # if dir.exists():
    #     shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir / 'labels'


def load_3d_info(anno_3D_path):
    with codecs.open(anno_3D_path, 'r', encoding='utf-8') as f:
        data_3D = json.load(f)

    if os.path.getsize(anno_3D_path) != len(json.dumps(data_3D).encode("utf-8")):
        print(subject, "smplx")
        import pdb; pdb.set_trace()

    results = {}

    for key, element in data_3D.items():
        element_key = key
        root_pose = element["smplx_param"]["root_pose"]
        body_pose = element["smplx_param"]["body_pose"]
        # lhand_pose = element["smplx_param"]["lhand_pose"]
        # rhand_pose = element["smplx_param"]["rhand_pose"]

        results[element_key] = {
                        "root_pose": root_pose,
                        "body_pose": body_pose,
                        # "lhand_pose": lhand_pose,
                        # "rhand_pose": rhand_pose
            }

    return results


def handle_one(subject):
    print('>>>>>>>>>> handling ', subject)
    anno_3D_path = os.path.join(anno_dir, subject, 'smplx_annotation.json')
    data_3D = load_3d_info(anno_3D_path)

    anno_path = os.path.join(anno_dir, subject, 'keypoint_annotation.json')
    with open(anno_path) as f:
        data = json.load(f)
        if os.path.getsize(anno_path) != len(json.dumps(data).encode("utf-8")):
            print(subject, "key")
            import pdb; pdb.set_trace()

    # Create image dict
    images_info = {'%g' % x['id']: x for x in data['images']}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    # imgToAnns = OrderedDict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    total_image_number = len(imgToAnns)
    print('>>>>>>>>>>>>total number of image is:', total_image_number)

    finger_ind = np.asarray([2, 4, 5, 8, 9, 12, 13, 16, 17, 20])
    for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {anno_path}'):
        img_info = images_info['%g' % img_id]
        h, w, pic_name = img_info['height'], img_info['width'], img_info['file_name']

        pic_name = subject + '-' + '-'.join(pic_name.split(os.sep))
        pic_path = os.path.join(img_root, "train", "train_ubody", pic_name)
        if not os.path.exists(pic_path):
            pic_path = os.path.join(img_root, "val", "val_ubody", pic_name)
            if not os.path.exists(pic_path):
                continue

        img_raw = cv2.imread(pic_path)
        if not isinstance(img_raw, np.ndarray):
            print('>>>>>>>>>>>>', pic_path, ' can not be loaded as numpy.numpy !')
            continue

        labels = []
        for ann in anns:
            # if ann['iscrowd'] or not ann['lefthand_valid'] or not ann['righthand_valid']:
            if ann['iscrowd']:
                continue

            try:
                data_3D[str(ann['id'])]
            except Exception as e:
                print(e)
                print(pic_path)
                print(ann['id'])
                with open('error.txt', 'a') as file:
                    file.write(pic_path + '\n')
                    file.write(str(ann['id']) + '\n')
                    file.write(str(e) + '\n')
                    file.write('\n')
                continue

            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = 0
            kpt = np.zeros([37 * 3], dtype=np.float32)
            k = (np.array(ann['keypoints']).reshape(-1, 3) / np.array([w, h, 1]))
            k[np.where(k[:, 2] == 0), :] = np.asarray([0, 0, 0])
            k = k.reshape(-1)
            kpt[:17 * 3] = k
            label = [cls] + box.tolist() + kpt.tolist()
            ann_3D = make_3D_ann(data_3D[str(ann['id'])], 'body')
            labels.append(label + ann_3D)

            if ann.get('lefthand_valid') is True:
                box = np.array(ann['lefthand_box'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                kpt = np.zeros([37 * 3], dtype=np.float32)
                kl = (np.array(ann['lefthand_kpts']).reshape(-1, 3)[finger_ind, :] / np.array([w, h, 1]))
                kl[np.where(kl[:, 2] == 0), :] = np.asarray([0, 0, 0])
                kl = kl.reshape(-1)
                kpt[51:51 + 30] = kl
                label = [1] + box.tolist() + kpt.tolist()
                ann_3D = make_3D_ann(data_3D[str(ann['id'])], 'lhand')
                labels.append(label + ann_3D)

            if ann.get('righthand_valid') is True:
                box = np.array(ann['righthand_box'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                kpt = np.zeros([37 * 3], dtype=np.float32)
                kr = (np.array(ann['righthand_kpts']).reshape(-1, 3)[finger_ind, :] / np.array([w, h, 1]))
                kr[np.where(kr[:, 2] == 0), :] = np.asarray([0, 0, 0])
                kr = kr.reshape(-1)
                kpt[81:] = kr
                label = [2] + box.tolist() + kpt.tolist()
                ann_3D = make_3D_ann(data_3D[str(ann['id'])], 'rhand')
                labels.append(label + ann_3D)

            if ann.get('keypoints') is not None:
                kpts_face = np.array(ann['keypoints']).reshape(-1, 3)[:5]
                kpts_face[np.where(kpts_face[:, 2] == 0), :] = np.asarray([0, 0, 0])
                # kpts_face = np.array(ann['keypoints']).reshape(-1, 3)[:5]
                musk_valid = kpts_face[:, 2] > 0
                if np.sum(musk_valid) > 2:
                    center = np.mean(kpts_face[musk_valid, :2], axis=0)
                    x_min = np.min(kpts_face[musk_valid, 0])
                    x_max = np.max(kpts_face[musk_valid, 0])
                    if x_min != x_max and (x_max - x_min) > min(w, h) / 32:
                        head_width = (x_max - x_min) * 1.2
                        head_height = head_width * 1.5

                        kpt = np.zeros([37 * 3], dtype=np.float32)
                        kpt[:5 * 3] = (kpts_face / np.array([w, h, 1])).reshape(-1)
                        label = [3] + [center[0] / w, center[1] / h, head_width / w, head_height / h] + kpt.tolist()
                        ann_3D = make_3D_ann(data_3D[str(ann['id'])], 'head')
                        labels.append(label + ann_3D)

        # Write
        # import pdb; pdb.set_trace()
        label_save_dir =  save_dir / Path(pic_path.split("images/")[-1])
        txt_save_path = os.path.join(os.path.dirname(label_save_dir), Path(pic_name).with_suffix('.txt'))
        if not os.path.exists(txt_save_path):
            with open(txt_save_path, 'a') as file:
                for i in range(len(labels)):
                    line = *(labels[i]),  # cls, box, keypoints
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


ubody_root_dir = r'/home/shou/work/3DAT/dataset/coco_ubody_3D'
anno_dir = os.path.join(ubody_root_dir, 'annotations')
img_root = os.path.join(ubody_root_dir, 'images')

save_dir = os.path.join(ubody_root_dir, 'labels')
save_dir = make_dirs(save_dir)

subject_list = [dI for dI in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, dI))]
subject_list.sort()


if __name__ == '__main__':
    for subject in subject_list:
        handle_one(subject)

    # pool = Pool(processes=2)
    # pool.map(handle_one, subject_list)
    # pool.close()
    # pool.join()

    end = time.time()

