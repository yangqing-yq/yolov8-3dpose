# full-body-capture-training
 This repository is body and hand train code for full-body mocap. 

## Installation
 H4W Installation 

## Downloading the datasets and weights
You can download the related pretrained weights from the [Baidu cloud disk](https://pan.baidu.com/s/14QgINWePQ-YlhkBh_L0KdQ?pwd=2oao)    
You can download the related datasets from the [Baidu cloud disk](https://pan.baidu.com/s/1ymgXh0JjS3-Ksgu3IXgQqA?pwd=8k0z)   
There can either be the path to the file with the parameters or a directory with the following structure:
```bash
models
├── data
│   ├── FreiHAND
│   │   ├── FreiHAND.py
│   │   ├── data
│   │   ├── evaluation
│   │   └── rootnet_output
│   ├── ...
│   └── ...
├── configs
│   ├── body.yaml
│   └── hand.yaml
├── weights(Download)
│   ├── body_pose_resnet_50_256x192.pth.tar
│   └── resnet50.pth
├── nets
├── utils
│   ├── human_model_files(Download)
│   └── smplx
├── base.py
├── model.py
├── test.py
└── train.py
```

## Train
    Train Body  
    python train.py --gpu 0 --parts body
    Train Hand
    python train.py --gpu 0 --parts hand

## Eval
    Test Body  
    python test.py --gpu 0 --parts body --test_epoch 6
    Train Hand
    python test.py --gpu 0 --parts hand --test_epoch 12

## References
[H4W](https://github.com/mks0601/Hand4Whole_RELEASE)  
