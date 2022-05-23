# Dualray:Dual-view Network for Prohibited Items Detection in X-ray Images
This code for the paper"Dual-view Network for Prohibited Items Detection in X-ray Images"  

## About Dataset
If you need our dataset, please contact our corresponding author and you will be required to sign a confidentiality agreement.  
Corresponding author email : zhg2018@sina.com  
### Dualray
We produced a dual-view dataset(Dualray), which includes a vertical view and a horizontal view. And there are corresponding annotations on each viewpoint.
![viewpoint](https://github.com/zhg-SZPT/Dualray/blob/main/img/viewpoint.png)

There are a total of 6 categories of prohibited items defined in the dataset, namely Knife, Wrench, pliers, scissors, powerbank, lighter. Each image is provided with annotation.


## DVNet Code
We propose the Dual-view Network(DVNet). The detection capability is improved by using the attention module to allow the network to learn the dependency relationships of objects in different perspective views.   
![framework](https://github.com/zhg-SZPT/Dualray/blob/main/img/framework.png)

### Dataset Preparation 
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

### Requirements
python 3.8  
pytorch 1.11.10  
cuda 11.2  
numpy  
matplotlib  
pillow  
opencv-python   

### Usage
