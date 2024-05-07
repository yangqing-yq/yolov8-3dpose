#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
生成标签数据
类别为4个类：全身、头、左手、右手
关键点取全身17点，左手10点，右手10点
crop 出单个人的图片，如果有重叠，合并后crop
'''


import os
import numpy as np
from pathlib import Path
import shutil
import json
from collections import defaultdict
import cv2
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
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
    left_column_max  = max(rec1[0],rec2[0])
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
    if 'keypoints' not in ann.keys():
        return None

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


def my_convert_coco(labels_dir='../coco/annotations/', save_dir='lable_save', img_root=''):
    save_dir = make_dirs(save_dir)  # output directory

    for json_file in sorted(Path(labels_dir).resolve().glob('*.json')):
        if 'val' in json_file.stem:
            label_save_dir = Path(save_dir) / 'labels' / 'val'/ json_file.stem
        elif 'train' in json_file.stem:
            label_save_dir = Path(save_dir) / 'labels' / 'train' / json_file.stem
        else:
            label_save_dir = Path(save_dir) / 'labels' / 'test' / json_file.stem
        label_save_dir.mkdir(parents=True, exist_ok=True)

        if 'val' in json_file.stem:
            images_save_dir = Path(save_dir) / 'images' / 'val' / json_file.stem
        elif 'train' in json_file.stem:
            images_save_dir = Path(save_dir) / 'images' / 'train' / json_file.stem
        else:
            images_save_dir = Path(save_dir) / 'images' / 'test' / json_file.stem
        images_save_dir.mkdir(parents=True, exist_ok=True)

        print('loading: ', json_file, '..................')
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images_info = {'%g' % x['id']: x for x in data['images']}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data['annotations']:
            imgToAnns[ann['image_id']].append(ann)

        if 'val' in json_file.stem:
            img_root_new = os.path.join(img_root, 'val2017')
        elif 'train' in json_file.stem:
            img_root_new = os.path.join(img_root, 'train2017')
        else:
            img_root_new = os.path.join(img_root, 'test2017')

        # Write labels file
        count = -1
        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            count = count + 1
            img_info = images_info['%g' % img_id]
            h, w, file_name = img_info['height'], img_info['width'], img_info['file_name']

            pic_path = os.path.join(img_root_new, file_name)
            img_raw = cv2.imread(pic_path)
            if not isinstance(img_raw, np.ndarray):
                print('>>>>>>>>>>>>',pic_path, ' can not be loaded as numpy.numpy !')
                continue

            tt = os.path.splitext(file_name)

            boxes_new = []
            anns_vaild = []
            for _, ann in enumerate(anns):
                labels = []
                # if ann['iscrowd'] or not ann['lefthand_valid'] or not ann['righthand_valid']:
                if 'iscrowd' not in ann.keys():
                    continue
                if ann['iscrowd']:
                    continue

                # if ann.get('lefthand_valid') is False and ann.get('righthand_valid') is False:
                #     continue

                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)

                cord_shift_x = box[2] * 0.1
                cord_shift_y = box[3] * 0.1

                x_start = np.clip(int(box[0] - cord_shift_x), 0, w)
                x_end = np.clip(int(box[0] + box[2] + cord_shift_x), 0, w)
                y_start = np.clip(int(box[1] - cord_shift_y), 0, h)
                y_end = np.clip(int(box[1] + box[3] + cord_shift_y), 0, h)

                boxes_new.append([x_start, y_start, x_end, y_end])
                anns_vaild.append(ann)
            boxes_new = np.asarray(boxes_new)

            if len(boxes_new)==0:
                continue

            # my_metric = np.asarray([[1, 1, 3, 3], [2, 2, 4, 4], [6, 6, 9, 9], [20, 20, 30, 30], [8, 8, 12, 12]])
            db_labels = DBSCAN(eps=0.5, min_samples=1, metric=my_metric).fit(boxes_new).labels_

            max_label = np.max(db_labels)
            order = 0
            for label in range(max_label + 1):
                ind = np.where(db_labels == label)[0]
                if len(ind) == 1:
                    ann = anns_vaild[ind[0]]
                    labels = []
                    # if ann['iscrowd'] or not ann['lefthand_valid'] or not ann['righthand_valid']:
                    if ann['iscrowd']:
                        continue

                    # if ann.get('lefthand_valid') is False and ann.get('righthand_valid') is False:
                    #     continue

                    x_start, y_start, x_end, y_end = boxes_new[ind[0]]

                    # 太小的图片直接放弃
                    if min(abs(x_start-x_end), abs(y_start-y_end)) < 128:
                        continue

                    img_new = img_raw[y_start:y_end, x_start:x_end]

                    labels = get_labels(ann, x_start, y_start, x_end, y_end)
                    if labels == None:
                        continue
                else:
                    boxes = boxes_new[ind]
                    x_start = np.min(boxes[:,0])
                    x_end = np.max(boxes[:,2])
                    y_start = np.min(boxes[:,1])
                    y_end = np.max(boxes[:,3])

                    # 太小的图片直接放弃
                    if min(abs(x_start - x_end), abs(y_start - y_end)) < 128:
                        continue

                    img_new = img_raw[y_start:y_end, x_start:x_end]
                    labels = []
                    for i in ind:
                        ann = anns_vaild[i]
                        if ann['iscrowd']:
                            continue

                        # if ann.get('lefthand_valid') is False and ann.get('righthand_valid') is False:
                        #     continue

                        labels_temp = get_labels(ann, x_start, y_start, x_end, y_end)
                        if labels_temp == None:
                            continue
                        if len(labels_temp)>0:
                            labels.extend(labels_temp)

                # 没有读出任何label，不必存储
                if len(labels) == 0:
                    continue

                pic_save_path = os.path.join(images_save_dir, tt[0] + '_{:0>3d}'.format(order) + tt[-1])
                cv2.imwrite(pic_save_path, img_new)

                txt_save_path = os.path.join(label_save_dir, tt[0] + '_{:0>3d}.txt'.format(order))
                order = order + 1
                # Write
                with open(txt_save_path, 'a') as file:
                    for i in range(len(labels)):
                        line = *(labels[i]),  # cls, box, keypoints
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')


img_root = r'/home/shou/work/3DAT/dataset/coco_wbody/unlabeled/images'
labels_dir = r'/home/shou/work/3DAT/dataset/coco_wbody/unlabeled/labels'
save_dir = r'/home/shou/work/3DAT/dataset/coco_wbody/labeled/'
my_convert_coco(labels_dir=labels_dir, save_dir = save_dir, img_root=img_root)


