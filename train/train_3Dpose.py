import os,sys

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
from ultralytics import YOLO


root_dir = os.path.dirname(os.path.dirname(__file__))
data_conf = os.path.join(root_dir, 'config', 'data_whole_det_37kpts_22euler.yaml')

weight_dir = ('/dc/mod_sifan/3dat-tracking/weights/')

def load_model(yaml_file, pt_file):
    if yaml_file and pt_file:
        model = YOLO(yaml_file, task = "pose").load(pt_file)
    elif yaml_file:
        model = YOLO(yaml_file, task = "pose")
    elif pt_file:
        model = YOLO(pt_file, task = "pose")
    else:
        print("Wrong input! No YAML nor pretrained file!")

    return model


if __name__ == '__main__':
    model_name = "yolov8n-3dpose"
    yaml_file = os.path.join(root_dir, 'config', model_name + '.yaml')

    pt_name = 'yolov8n-pose'
    pt_file = os.path.join(weight_dir, pt_name + '.pt')

    model = load_model(yaml_file, pt_file)

    # model.train(data = data_conf, epochs = 100, imgsz = 640, device = [1, 3, 5, 6, 7], batch = 32*5, name = "3dat")
    model.train(data = data_conf, epochs = 100, imgsz = 640, device = [0], batch = 32*4, name = "3dat")

    model.export()

    print('finish all!')
