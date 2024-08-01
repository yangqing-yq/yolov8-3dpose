# Installation

## Conda
    • conda create -n yolo3d python=3.8
    • conda activate yolo3d
    • conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
    • conda install -c conda-forge openvino=2024.2.0

## Environment
    # Install other required libraries
    # Tips, upgrade openvino-dev, torch, torchvision to ensure they are compatible
    • pip install -r requirements.txt


## Train(ongoing)
    python train/train_3Dpose.py

## Inference to pose param
    python infer/predict2json.py

## Inference to mesh
    python infer_render_3dmesh.py


## Evaluate to csv
    python sim_eval_3dmesh.py
    python combine_csvs.py