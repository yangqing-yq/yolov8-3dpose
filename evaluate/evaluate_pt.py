import os
from ultralytics import YOLO
import torch

import sys
sys.path.insert(0, sys.path[0]+"/../")
from tools.save_metrics import saveMetric

root_dir = os.path.dirname(os.path.dirname(__file__))
data_conf = os.path.join(root_dir, 'config', 'data_whole_det_37kpts_52euler.yaml')

model_root_dir = 'runs/pose/3dat6/weights'
model_name = 'best'



if __name__ == '__main__':
    dDevice = "gpu"

    # Load model
    model_path = os.path.join(model_root_dir, model_name + '.pt')
    if dDevice == "cpu":
        device = torch.device('cpu')
        model = YOLO(model_path).to(device)
    else:
        model = YOLO(model_path)

    # Validate
    metrics = model.val(data=data_conf, batch=16)

    # Save results
    # saveMetric(metrics, f"metrics_pytorch_{model_name}_{dDevice}")

