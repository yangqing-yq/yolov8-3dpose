import os
import sys
from ultralytics import YOLO

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_dir)
if __package__:
    from .det_api import DetBodyHandHead
else:
    from det_api import DetBodyHandHead
sys.path.pop()


root_dir = os.path.dirname(os.path.dirname(__file__))


def save_to_2d(bbox_body, bbox_lhand, bbox_rhand, bbox_head, kpts, src_path):
    pass

def save_to_3d(kpt_3dbody, file_name):
    kpt_3dbody_str = " ".join(map(str, kpt_3dbody))

    txt_path = os.path.join(root_dir, "infer/pred", file_name + ".txt")
    with open(txt_path, "w") as f:
        f.write(kpt_3dbody_str)


def save_txt(result, file_name):
    bbox_body, bbox_lhand, bbox_rhand, bbox_head, [kpt_body, kpt_lhand, kpt_rhand], [kpt_3dbody] = result

    # save_to_2d(bbox_body, bbox_lhand, bbox_rhand, bbox_head, [kpt_body, kpt_lhand, kpt_rhand], src_path)
    save_to_3d(kpt_3dbody, file_name)


def infer_one(src_path, model_path):
    model = DetBodyHandHead(model_path=model_path)

    person_bbox = []
    result = model(src_path, person_bbox)

    file_name = os.path.basename(src_path).split(".")[0]
    save_txt(result, file_name)


if __name__ == "__main__":
    mode = 1
    dataset_path = '/mnt/sh_flex_storage/home/shou/3DAT/AitPoseChecker3DAT/images/ubody/train/images'
    model_path = os.path.join(root_dir, 'runs/pose', 'augment2', 'weights/last.pt')

    if mode == 0: # one image
        src_path = os.path.join(dataset_path, "")
        infer_one(src_path)
    elif mode == 1: # dir
        for root, directories, files in os.walk(dataset_path):
            for file in files:
                src_path = os.path.join(root, file)
                infer_one(src_path, model_path)
    else:
        print("Other mode")