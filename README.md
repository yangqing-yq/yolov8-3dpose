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

## Dataset structure
```
${DATA_ROOT}  
|-- images 
|   |-- train
|   |   |-- train_ubody
|   |-- val
|   |   |-- val_ubody
|-- box
|   |-- train
|   |   |-- train_ubody
|   |-- val
|   |   |-- val_ubody
|-- 2d_kps
|   |-- train
|   |   |-- train_ubody
|   |-- val
|   |   |-- val_ubody
|-- 3d_angles
|   |-- train
|   |   |-- train_ubody
|   |-- val
|   |   |-- val_ubody
|-- shape
|   |-- train
|   |   |-- train_ubody
|   |-- val
|   |   |-- val_ubody
```

## Project structure
```
${PROJECT_ROOT}  
|-- ultralytics
|-- body_tracking
|   |-- weights
|   |-- utils
|   |   |-- human_model_files
|   |   |   |-- smpl
|   |   |   |   |-- SMPL_NEUTRAL.pkl
|   |   |   |   |-- SMPL_MALE.pkl
|   |   |   |   |-- SMPL_FEMALE.pkl
|   |   |   |-- smplx
|   |   |   |   |-- MANO_SMPLX_vertex_ids.pkl
|   |   |   |   |-- SMPL-X__FLAME_vertex_ids.npy
|   |   |   |   |-- SMPLX_NEUTRAL.pkl
|   |   |   |   |-- SMPLX_to_J14.pkl
|   |   |   |   |-- SMPLX_NEUTRAL.npz
|   |   |   |   |-- SMPLX_MALE.npz
|   |   |   |   |-- SMPLX_FEMALE.npz
|   |   |   |-- mano
|   |   |   |   |-- MANO_LEFT.pkl
|   |   |   |   |-- MANO_RIGHT.pkl
|   |   |   |-- flame
|   |   |   |   |-- flame_dynamic_embedding.npy
|   |   |   |   |-- flame_static_embedding.pkl
|   |   |   |   |-- FLAME_NEUTRAL.pkl
```

## Train(ongoing)
    python train/train_3Dpose.py

## Inference to pose param
    python infer/predict2json.py

## Inference to mesh
    python infer_render_3dmesh.py


## Evaluate to csv
    python sim_eval_3dmesh.py
    python combine_csvs.py