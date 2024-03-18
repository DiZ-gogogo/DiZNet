# DiZNet
DiZNet :An End-to-End Text Detection and Recognition Algorithm with Detail in text Zone

##### Introduction

This repository is the official implementation of DiZNet.

##### Installation

```shell
1、https://github.com/DiZ-gogogo/DiZNet.git
2、conda install pytorch torchvision -c pytorch
3、pip install -r requirement.txt
# build pa and other post-processing algorithms
sh ./compile.sh
```

##### Prepare datasets

```
DiZNet
└── data
    ├── CTW1500
    │   ├── train
    │   │   ├── text_image
    │   │   └── text_label_curve
    │   └── test
    │       ├── text_image
    │       └── text_label_curve
    ├── total_text
    │   ├── Images
    │   │   ├── Train
    │   │   └── Test
    │   └── Groundtruth
    │       ├── Polygon
    │       └── Rectangular
    ├── ICDAR2015
    │   └── Challenge4
    │       ├── ch4_training_images
    │       ├── ch4_training_localization_transcription_gt
    │       ├── ch4_test_images
    │       └── ch4_test_localization_transcription_gt
    ├── MSRA-TD500
    │   ├── train
    │   └── test
    ├── HUST-TR400
    ├── COCO-Text
    │   └── train2014
    ├── SynthText
    │   ├── 1
    │   ├── ...
    │   └── 200
    └── ICDAR2017-MLT
    │   ├── ch8_training_images
    │   ├── ch8_validation_images
    │   ├──  ch8_training_localization_transcription_gt_v2
    │   ├── ch8_validation_localization_transcription_gt_v2
    └── RCTW-17
        ├── train
```

##### Training & Testing

###### ICDAR 2015 Detection

In train_ic15.py, you can finely adjust model parameters, image size, batch size, learning rate, and other information. The script train_ic15.py allows generating model weights in the checkpoints directory. During testing, you can use the --resume option to load model parameters for testing.

```shell
# Training
python train_ic15.py --arch resnet18 --img_size 736 --short_size 736
# Test
python test_ic15.py --resume release_models/DiZNet_ic15_resnet18_736.pth.tar
```

###### ICDAR 2015 End-to-End Recognition

```shell
# Training
python pretrain.py --dataset joint --arch resnet18 --with_rec True --epoch 3 --img_size 736 --short_size 736
# Test
python test_ic15.py --resume release_models/DiZNet_joint_resnet18_736_with_rec.pth.tar --with_rec True --min_score 0.8 --rec_ignore_score 0.93
```

For training text detection and recognition on the Total-Text dataset, run train_tt.py; for testing, run test_tt.py.

