# Dualray:Dual-view Network for Prohibited Items Detection in X-ray Images
This code for the paper"Dual-view Network for Prohibited Items Detection in X-ray Images"

## About Dataset
If you need our dataset, please contact our corresponding author and you will be required to sign a confidentiality agreement
Corresponding author email : zhg2018@sina.com

## Requirements
python 3.8
pytorch 1.11.10
cuda 11.2

## Dataset Preparation
1. This article uses .txt format for training. Dualray will provide the annotation in .xml format. Before training, you need to run voc_annotation.py to generate a tag file in .txt format.
2. Your ~/Dualray/ folder should look like this
'''
Dualray
├── X_ray2022/
│   └── ...
└── X_ray2022_1/ 
    └── ...
'''
