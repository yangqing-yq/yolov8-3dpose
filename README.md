# Installation

## Conda
    • conda create -n yolov8 python=3.8
    • conda activate yolov8
    • conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    • conda install -c conda-forge openvino=2023.2.0

## Environment
    # Install other required libraries
    • pip install ultralytics
    # Tips, upgrade openvino-dev, torch, torchvision to ensure they are compatible
    • pip install onnx tensorrt 
    • pip install --upgrade openvino-dev
    • pip install --upgrade torch
    • pip install --upgrade torchvision
    • conda install opencv=4.9.0

## Install nncf
    • git clone git@github.com:openvinotoolkit/nncf.git
    • cd nncf
    • pip install .

# Train and infer
    python train/train_3Dpose.py
    python infer/predict2json.py