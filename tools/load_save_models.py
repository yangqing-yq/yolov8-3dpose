import os
from pathlib import Path


def load_ov_model(model_root_dir, model_name, data_type, compress = ''):
    import openvino.runtime as ov
    ov_path = Path(f"{model_root_dir}/{model_name}_{data_type}_openvino_model{compress}/{model_name}.xml")
    ov_model = ov.Core().read_model(ov_path)

    return ov_model, ov_path


def save_ov_model(model, model_root_dir, model_name, data_type, compress = ''):
    import openvino.runtime as ov
    save_path = os.path.join(model_root_dir, model_name + f'_{data_type}_openvino_model{compress}', 'model.xml')
    ov.save_model(model, save_path)


def load_pt_model(model_root_dir, model_name, compress = '', data_type = 'fp32'):
    import torch
    if compress == '':
        pt_path = os.path.join(model_root_dir, model_name + '.pt')
    else:
        Path(f"{model_root_dir}/{model_name}_pytorch_model{compress}/{model_name}_{data_type}.pt")

    from ultralytics import YOLO
    pt_model = YOLO(pt_path)

    return pt_model, pt_path


def save_pt_model(model, model_root_dir, model_name, data_type, compress = ''):
    import torch
    save_path = Path(f"{model_root_dir}/{model_name}_pytorch_model{compress}/{model_name}_{data_type}.pt")
    torch.save (model, save_path)