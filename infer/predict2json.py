import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

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


def infer_one(src_path, model_path):
    model = DetBodyHandHead(model_path=model_path)

    person_bbox = []
    result = model(src_path, person_bbox)
    bbox_body, bbox_lhand, bbox_rhand, bbox_head, [kpt_body, kpt_lhand, kpt_rhand], [kpt_3dbody] = result

    return {'root_pose':kpt_3dbody[0:3] , 'body_pose':kpt_3dbody[-63:]}

def process_file(src_path, model_path):
    kpt_3dbody = infer_one(src_path, model_path)
    file_name = os.path.basename(src_path)
    return file_name, kpt_3dbody

if __name__ == "__main__":
    mode = 1
    train_or_val = "train"
    dataset_path = f'/data/coco_human/images/{train_or_val}/{train_or_val}_ubody'
    print("dataset_path:", dataset_path)
    model_path = os.path.join(root_dir, 'runs/pose', '3dat19', 'weights/last.pt')
    print("model_path:",model_path)
    if mode == 0: # one image
        src_path = os.path.join(dataset_path, "ConductMusic-YT-ConductMusic_S34_Trim3-ConductMusic_S34_Trim3_scene002-000101.png")
        src_path = "test.png"
        res= infer_one(src_path,model_path)
        print("root_pose:",len(res['root_pose']))
        print("body_pose:",len(res['body_pose']))
    elif mode == 1: # dir
        pred_3dpose = {}
        with ProcessPoolExecutor(max_workers=10) as executor:
            future_to_file = {executor.submit(process_file, os.path.join(root, file), model_path): file for root, _, files in os.walk(dataset_path) for file in files}
            for future in as_completed(future_to_file):
                print("future:",future)
                file_name, kpt_3dbody = future.result()
                pred_3dpose[file_name] = kpt_3dbody
                print("kpt_3dbody:",kpt_3dbody)
                # save_to_3d(kpt_3dbody, file_name)
        # for root, directories, files in os.walk(dataset_path):
        #     for file in files:
        #         src_path = os.path.join(root, file)
        #         kpt_3dbody = infer_one(src_path, model_path)

        #         file_name = os.path.basename(src_path).split("/")[-1]
        #         pred_3dpose[file_name] = kpt_3dbody
        #         # save_to_3d(kpt_3dbody, file_name)
        print("json")
        with open(f'pred_{train_or_val}_noaug.json', 'w') as f:
            json.dump(pred_3dpose, f,  indent=4)
    else:
        print("Other mode")