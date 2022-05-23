# Dualray:Dual-view Network for Prohibited Items Detection in X-ray Images
This code for the paper"Dual-view Network for Prohibited Items Detection in X-ray Images"  

## About Dataset
If you need our dataset, please contact our corresponding author and you will be required to sign a confidentiality agreement.  
Corresponding author email : zhg2018@sina.com  
### Dualray


## Requirements
python 3.8  
pytorch 1.11.10  
cuda 11.2  
numpy  
matplotlib  
pillow  
opencv-python  

## DVNet Code


## Dataset Preparation 
1. Our code needs to use images from two different perspectives. The vertical view images is in X_ray2022 and the horizontal view images is in X_ray2022_1.   
Your `~/Dualray/` folder should look like this  
```
Dualray
├── X_ray2022/
│   └── ...
└── X_ray2022_1/
    └── ...
```
2. This article uses .txt format for training. Dualray will provide the annotation in .xml format. Before training, you need to run voc_annotation.py to generate a tag file in .txt format. 
