import os,sys
import numpy as np
import cv2
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))

from ultralytics import YOLO


cur_dir = os.path.abspath(os.path.dirname(__file__))


def compute_IOU(rec1,rec2):
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
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        # S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        # return S_cross/(S1+S2-S_cross)
        return S_cross / S1


def filter_box(bbox_person, bboxes_part, kpts_part):
    # 根据置信度和bbox重合度过滤
    # if is_right:
    #     kpt_wrist = kpt_person[10]
    # else:
    #     kpt_wrist = kpt_person[9]

    bbox_valid = []
    kpt_valid = []
    conf_max = 0
    for ind in range(len(bboxes_part)):
        bbox = bboxes_part[ind]
        kpt = kpts_part[ind]
        conf = bbox[4]
        cls = int(bbox[5])
        if cls == 0 or conf < 0.3:
            continue

        ratio = compute_IOU(bbox[:4], bbox_person[:4])
        if ratio > 0.4 and conf > conf_max:
            conf_max = conf
            bbox_valid = bbox[:4]
            kpt_valid = kpt

    return bbox_valid, kpt_valid


def filter_box_new(bbox_person, bboxes_hand, kpts_part, kpt_wrist):
    # 根据置信度和bbox重合度过滤
    bbox_valid = []
    kpt_valid = []

    if kpt_wrist[2] < 0.3:
        return bbox_valid, kpt_valid

    dis_min = 320
    for ind in range(bboxes_hand.shape[0]):
        bbox = bboxes_hand[ind]
        kpt = kpts_part[ind]
        conf = bbox[4]
        cls = int(bbox[5])
        if cls == 0 or conf < 0.3:
            continue

        edge_ave = ((bbox[2]-bbox[0]) + (bbox[3]-bbox[1]))*0.5
        center = np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        dis = np.linalg.norm(center - kpt_wrist[:2])
        if dis > edge_ave*1.2:
            continue

        ratio = compute_IOU(bbox[:4], bbox_person[:4])
        if ratio > 0.2 and dis < dis_min:
            dis_min = dis
            bbox_valid = bbox[:4]
            kpt_valid = kpt

    return bbox_valid, kpt_valid


def process_bbox_head(kpts, img):

    shape = img.shape[:2]

    kpts_face = kpts[:5]
    center = np.mean(kpts_face[:,:2], axis=0)
    x_min = np.min(kpts_face[:, 0])
    x_max = np.max(kpts_face[:, 0])
    head_width_half = (x_max - x_min) * 0.6
    head_height_half = head_width_half * 1.5

    pt1_x = np.clip(center[0]-head_width_half, 0, shape[1])
    pt1_y = np.clip(center[1]-head_height_half, 0, shape[0])
    pt2_x = np.clip(center[0]+head_width_half, 0, shape[1])
    pt2_y = np.clip(center[1]+head_height_half, 0, shape[0])

    bbox_head = np.asarray([pt1_x, pt1_y, pt2_x, pt2_y])

    return bbox_head


def match_wrist(bbox, kpt1, kpt2, kpt_person):
    center = np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    if kpt1[2] < 0.5:
        dis1 = 10000
    else:
        dis1 = np.linalg.norm(center - kpt1[:2])

    if kpt2[2] < 0.5:
        dis2 = 10000
    else:
        dis2 = np.linalg.norm(center - kpt2[:2])

    # 左右手手腕的距离
    if kpt1[2] < 0.5 or kpt2[2] < 0.5:
        dis_l_r = 10000
    else:
        dis_l_r = np.linalg.norm(kpt1[:2] - kpt2[:2])
    # 设置手腕间距阈值
    if kpt_person[5][2] > 0.5 and kpt_person[6][2] > 0.5:
        th = np.linalg.norm(kpt_person[5][:2] - kpt_person[6][:2]) / 2
    elif kpt_person[3][2] > 0.5 and kpt_person[4][2] > 0.5:
        th = np.linalg.norm(kpt_person[3][:2] - kpt_person[4][:2])
    elif kpt_person[1][2] > 0.5 and kpt_person[2][2] > 0.5:
        th = np.linalg.norm(kpt_person[1][:2] - kpt_person[2][:2])*2
    else:
        th = min((abs(bbox[0] - bbox[2]), abs(bbox[1] - bbox[3]))) / 4

    if dis1 > dis2 and dis_l_r > th:
        res = False
    else:
        res = True

    return res


def filter_box_cover(lhand_bbox, kpt_lh, rhand_bbox, kpt_rh, kpt_person):
    ratio = compute_IOU(lhand_bbox, rhand_bbox)
    if ratio > 0.6:
        kpt_lhand = kpt_person[9]
        kpt_rhand = kpt_person[10]
        if not match_wrist(lhand_bbox, kpt_lhand, kpt_rhand, kpt_person):
            lhand_bbox, kpt_lh = [], []
        elif not match_wrist(rhand_bbox, kpt_rhand, kpt_lhand, kpt_person):
            rhand_bbox, kpt_rh = [], []

    return lhand_bbox, kpt_lh, rhand_bbox, kpt_rh


def get_hand_bbox(bboxes, cls, kpts, bbox_body, kpt_body, type='left'):
    if type == 'left':
        ind_hand_mask = cls == 1
        kpts_hand = kpts[ind_hand_mask][:, 17:27]
        kpt_wrist = kpt_body[9]
    else:
        ind_hand_mask = cls == 2
        kpts_hand = kpts[ind_hand_mask][:, 27:]
        kpt_wrist = kpt_body[10]
    bboxes_hand = bboxes[ind_hand_mask]

    bbox_hand, kpt_hand = filter_box_new(bbox_body, bboxes_hand, kpts_hand, kpt_wrist)

    return bbox_hand, kpt_hand


class DetBodyHandHead():
    def __init__(self, model_path):
        self.model = YOLO(model_path, task='pose')
        self.error_count = 0
        self.redetect_th = 3
        self.bbox_old = []
        # self.model.export(format='engine', device=0, half=True)

    def __call__(self, img, bbox_pre):

        # results = self.model(img, verbose=False)
        results = self.model(img, half=True, verbose=False)
        # results = self.model(img)

        bboxes = results[0].boxes.data.cpu().numpy()

        bbox_body, bbox_lhand, bbox_rhand, bbox_head = [], [], [], []
        kpt_body, kpt_lhand, kpt_rhand = [], [], []
        kpt_3dbody = []
        self.error_count = self.error_count + 1
        if len(bboxes) == 0:
            pass
        else:
            cls = np.asarray(bboxes[:,-1], dtype=np.int16)
            ind_body_mask = cls == 0
            bboxes_body = bboxes[ind_body_mask]
            if len(bboxes_body)==0:
                pass
            else:
                self.error_count = 0
                # >>>>>>>>>>>>>>> body
                kpts = results[0].keypoints.data.cpu().numpy()
                kpts_person = kpts[ind_body_mask][:, :17]
                kpts_3d = results[0].keypoints_3d.cpu().numpy()
                kpts_3dperson = kpts_3d[ind_body_mask][:, :22]
                if len(bbox_pre)==0 and len(self.bbox_old)==0:
                    # 选取面积最大的box
                    areas_person = (bboxes_body[:, 2] - bboxes_body[:, 0]) * (bboxes_body[:, 3] - bboxes_body[:, 1])
                    ind = np.argmax(areas_person)
                    bbox_body = bboxes_body[ind][:4]
                    kpt_body = kpts_person[ind]
                    kpt_3dbody = kpts_3dperson[ind].flatten().tolist()
                    self.bbox_old = bbox_body
                else:
                    if len(bbox_pre)!=0:
                        self.bbox_old = bbox_pre
                    ratio_max = -1
                    for ind in range(bboxes_body.shape[0]):
                        bbox_new = bboxes_body[ind][:4]
                        ratio = compute_IOU(self.bbox_old[:4], bbox_new[:4])
                        if ratio > ratio_max:
                            ratio_max = ratio
                            bbox_body = bbox_new
                            kpt_body = kpts_person[ind]
                            kpt_3dbody = kpts_3dperson[ind].flatten().tolist()

                # >>>>>>>>>>>>>>> left hand
                bbox_lhand, kpt_lhand = get_hand_bbox(bboxes, cls, kpts, bbox_body, kpt_body, type='left')

                # >>>>>>>>>>>>>>> right hand
                bbox_rhand, kpt_rhand = get_hand_bbox(bboxes, cls, kpts, bbox_body, kpt_body, type='right')

                # >>>>>>>>>>>>>>> head
                # method1,box帧间不平滑
                # ind_head_mask = cls == 3
                # boxes_head = boxes[ind_head_mask]
                # if len(boxes_head) == 0:
                #     bbox_head = []
                # else:
                #     bbox_head = filter_box(bbox_body, boxes_head)
                # method2
                img_ori = results[0].orig_img
                bbox_head = process_bbox_head(kpt_body, img_ori)

        if self.error_count > self.redetect_th:
            self.bbox_old = []

        return bbox_body, bbox_lhand, bbox_rhand, bbox_head, [kpt_body, kpt_lhand, kpt_rhand], [kpt_3dbody]



if __name__ == "__main__":
    model_path = os.path.join(cur_dir, 'models', 'best_x.pt')
    if not os.path.exists(model_path):
        raise Exception('error!  does not exists: {}'.format(model_path))
    model = DetBodyHandHead(model_path = model_path)

    pic_path = os.path.join(cur_dir, 'test_pic', '00012.png')

    img = cv2.imread(pic_path)
    model(img)

    print('finish all!')
