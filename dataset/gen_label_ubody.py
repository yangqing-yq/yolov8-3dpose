'''
生成标签数据
类别为4个类：全身、头、左手、右手
关键点取全身17点，左手10点，右手10点
'''


import numpy as np
from pathlib import Path
import shutil
import json
from collections import defaultdict
from collections import OrderedDict
import cv2
from tqdm import tqdm
import os
import time
from multiprocessing import Pool
from sklearn.cluster import DBSCAN



def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    # if dir.exists():
    #     shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels', dir / 'images':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def judge_no_overlap(rec1,rec2):
    """
    计算两个矩形框的交集除以其中一个矩形的面积。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max = max(rec1[1],rec2[1])
    down_row_min = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 1
    # 两矩形有相交区域的情况
    else:
        return 0


def my_metric(x, y):
    return judge_no_overlap(x, y)


def get_labels(ann, x_start, y_start, x_end, y_end):
    finger_ind = np.asarray([2, 4, 5, 8, 9, 12, 13, 16, 17, 20])

    labels = []
    w_new = x_end - x_start
    h_new = y_end - y_start

    # The COCO box format is [top left x, top left y, width, height]
    box = np.array(ann['bbox'], dtype=np.float64)

    box[0] = box[0] - x_start
    box[1] = box[1] - y_start

    box[:2] += box[2:] / 2  # xy top-left corner to center
    box[[0, 2]] /= w_new  # normalize x
    box[[1, 3]] /= h_new  # normalize y
    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
        return []

    kpt = np.zeros([37 * 3], dtype=np.float32)
    k = np.array(ann['keypoints']).reshape(-1, 3) - np.array([x_start, y_start, 0])
    k = k / np.array([w_new, h_new, 1])
    k[np.where(k[:, 2] == 0), :] = np.asarray([0, 0, 0])
    k = k.reshape(-1)
    kpt[:17 * 3] = k
    label = [0] + box.tolist() + kpt.tolist()
    labels.append(label)

    for i in range(1):
        if ann.get('lefthand_valid') is True:
            box = np.array(ann['lefthand_box'], dtype=np.float64)
            box[0] = box[0] - x_start
            box[1] = box[1] - y_start
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w_new  # normalize x
            box[[1, 3]] /= h_new  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            kpt = np.zeros([37 * 3], dtype=np.float32)
            kl = np.array(ann['lefthand_kpts']).reshape(-1, 3)[finger_ind, :] - np.array(
                [x_start, y_start, 0])
            kl = kl / np.array([w_new, h_new, 1])
            kl[np.where(kl[:, 2] == 0), :] = np.asarray([0, 0, 0])
            kl = kl.reshape(-1)
            kpt[51:51 + 30] = kl
            label = [1] + box.tolist() + kpt.tolist()
            labels.append(label)

    for i in range(1):
        if ann.get('righthand_valid') is True:
            box = np.array(ann['righthand_box'], dtype=np.float64)
            box[0] = box[0] - x_start
            box[1] = box[1] - y_start
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w_new  # normalize x
            box[[1, 3]] /= h_new  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            kpt = np.zeros([37 * 3], dtype=np.float32)
            kr = np.array(ann['righthand_kpts']).reshape(-1, 3)[finger_ind, :] - np.array(
                [x_start, y_start, 0])
            kr = kr / np.array([w_new, h_new, 1])
            kr[np.where(kr[:, 2] == 0), :] = np.asarray([0, 0, 0])
            kr = kr.reshape(-1)
            kpt[81:] = kr
            label = [2] + box.tolist() + kpt.tolist()
            labels.append(label)

    if ann.get('keypoints') is not None:
        kpts_face = np.array(ann['keypoints']).reshape(-1, 3)[:5]
        kpts_face = kpts_face - np.array([x_start, y_start, 0])
        kpts_face[np.where(kpts_face[:, 2] == 0), :] = np.asarray([0, 0, 0])
        # kpts_face = np.array(ann['keypoints']).reshape(-1, 3)[:5]
        musk_valid = kpts_face[:, 2] > 0
        if np.sum(musk_valid) > 2:
            center = np.mean(kpts_face[musk_valid, :2], axis=0)
            x_min = np.min(kpts_face[musk_valid, 0])
            x_max = np.max(kpts_face[musk_valid, 0])
            if x_min != x_max and (x_max - x_min) > min(w_new, h_new) / 16:
                head_width = (x_max - x_min) * 1.2
                head_height = head_width * 1.5

                kpt = np.zeros([37 * 3], dtype=np.float32)
                kpt[:5 * 3] = (kpts_face / np.array([w_new, h_new, 1])).reshape(-1)
                label = [3] + [center[0] / w_new, center[1] / h_new, head_width / w_new,
                               head_height / h_new] + kpt.tolist()
                labels.append(label)

    return labels


def handle_one(subject):
    print('>>>>>>>>>> handling ', subject)
    anno_path = os.path.join(anno_dir, subject, 'keypoint_annotation.json')

    with open(anno_path) as f:
        data = json.load(f)

    # Create image dict
    images_info = {'%g' % x['id']: x for x in data['images']}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    # imgToAnns = OrderedDict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    total_image_number = len(imgToAnns)
    print('>>>>>>>>>>>>total number of image is:', total_image_number)
    val_ind = int(total_image_number*0.8)

    count = -1
    is_train = True
    fqc = 10

    images_save_dir = train_images_save_dir
    label_save_dir = train_label_save_dir

    finger_ind = np.asarray([2, 4, 5, 8, 9, 12, 13, 16, 17, 20])
    for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {anno_path}'):
        count = count + 1
        if count>val_ind:
            is_train = False
            fqc = 45
            images_save_dir = val_images_save_dir
            label_save_dir = val_label_save_dir

        # 每fqc帧取一个样本
        if img_id % fqc != 0:
            continue

        img_info = images_info['%g' % img_id]
        h, w, pic_name = img_info['height'], img_info['width'], img_info['file_name']

        pic_path = os.path.join(img_root, subject, pic_name)
        img_raw = cv2.imread(pic_path)
        if not isinstance(img_raw, np.ndarray):
            print('>>>>>>>>>>>>', pic_path, ' can not be loaded as numpy.numpy !')
            continue

        pic_name_new = subject + '-' + '-'.join(pic_name.split(os.sep))
        tt = os.path.splitext(pic_name_new)

        labels = []
        for ann in anns:
            # if ann['iscrowd'] or not ann['lefthand_valid'] or not ann['righthand_valid']:
            if ann['iscrowd']:
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
            labels.append(label)

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
                labels.append(label)

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
                labels.append(label)

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
                        labels.append(label)

        # Write
        pic_save_path = os.path.join(images_save_dir, pic_name_new)
        shutil.copy(pic_path, pic_save_path)

        txt_save_path = os.path.join(label_save_dir, Path(pic_name_new).with_suffix('.txt'))
        with open(txt_save_path, 'a') as file:
            for i in range(len(labels)):
                line = *(labels[i]),  # cls, box, keypoints
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


ubody_root_dir = r'/home/mmzhu/data2/datasets_human/Ubody'
anno_dir = os.path.join(ubody_root_dir, 'annotations')
img_root = os.path.join(ubody_root_dir, 'images')
video_dir = os.path.join(ubody_root_dir, 'videos')

save_dir = 'ubody_data'
save_dir = make_dirs(save_dir)

train_label_save_dir = Path(save_dir) / 'labels' / 'train' / 'train_ubody'
train_label_save_dir.mkdir(parents=True, exist_ok=True)
train_images_save_dir = Path(save_dir) / 'images' / 'train' / 'train_ubody'
train_images_save_dir.mkdir(parents=True, exist_ok=True)

val_label_save_dir = Path(save_dir) / 'labels' / 'val' / 'val_ubody'
val_label_save_dir.mkdir(parents=True, exist_ok=True)
val_images_save_dir = Path(save_dir) / 'images' / 'val' / 'val_ubody'
val_images_save_dir.mkdir(parents=True, exist_ok=True)

subject_list = [dI for dI in os.listdir(anno_dir) if os.path.isdir(os.path.join(anno_dir, dI))]
subject_list.sort()

subject_list = subject_list[:1]

print(subject_list)


if __name__ == '__main__':
    start = time.time()
    for subject in subject_list:
        handle_one(subject)

    # pool = Pool(processes=2)
    # pool.map(handle_one, subject_list)
    # pool.close()
    # pool.join()

    end = time.time()
    print('total time is: {} s'.format(end - start))

    print('finish all!')

