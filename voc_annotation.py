import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode=0      Randomly obtain a training set and a test set that can be used for training
#   annotation_mode=1      Get TXT in imagesets
#   annotation_mode=2      Get the training set in imageset
#--------------------------------------------------------------------------------------------------------------------------------#
# annotation_mode     = 0
annotation_mode     = 2

classes_path        = 'model_data/xray_classes.txt'
# trainval_percent    = 0.9
# train_percent       = 0.9
trainval_percent    = 0.8
train_percent       = 0.8

# VOCdevkit_path  = '/vol/mordy/X_raydevkit'
VOCdevkit_path  = 'Dualray'
# VOCdevkit_path  = '../X_raydevkit'
# VOCdevkit_path  = './X_raydevkit_1'

VOCdevkit_sets  = [('2022', 'train'), ('2022', 'val')]
VOCdevkit_sets_1 =[('2022', 'train_1'), ('2022', 'val_1')]
classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'X_ray%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
def convert_annotation_1(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'X_ray%s_1/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")

        xmlfilepath     = os.path.join(VOCdevkit_path, 'X_ray2022/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'X_ray2022/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        temp_xml.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        xmlfilepath_1  = os.path.join(VOCdevkit_path, 'X_ray2022_1/Annotations')
        saveBasePath_1 = os.path.join(VOCdevkit_path, 'X_ray2022_1/ImageSets/Main')
        temp_xml_1     = os.listdir(xmlfilepath_1)
        temp_xml_1.sort(key=lambda x: int(x.split("_")[-2]))
        total_xml_1    = []
        for xml in temp_xml_1:
            if xml.endswith(".xml"):
                total_xml_1.append(xml)

        num     = len(total_xml)
        num_1   = len(total_xml_1)
        if num != num_1:
            print('please check the number of dataset')
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)
                if i in train:  
                    ftrain.write(name)
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in view1 ImageSets done.")

        ftrainval_1 = open(os.path.join(saveBasePath_1, 'trainval_1.txt'), 'w')
        ftest_1 = open(os.path.join(saveBasePath_1, 'test_1.txt'), 'w')
        ftrain_1 = open(os.path.join(saveBasePath_1, 'train_1.txt'), 'w')
        fval_1 = open(os.path.join(saveBasePath_1, 'val_1.txt'), 'w')

        for i in list:
            name = total_xml_1[i][:-4] + '\n'
            if i in trainval:
                ftrainval_1.write(name)
                if i in train:
                    ftrain_1.write(name)
                else:
                    fval_1.write(name)
            else:
                ftest_1.write(name)

        ftrainval_1.close()
        ftrain_1.close()
        fval_1.close()
        ftest_1.close()
        print("Generate txt in view2 ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2022_train.txt and 2022_val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'X_ray%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/X_ray%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate 2022_train.txt and 2022_val.txt for train done.")

        print("Generate 2022_train_1.txt and 2022_val_1.txt for train.")
        for year, image_set in VOCdevkit_sets_1:
            image_ids = open(os.path.join(VOCdevkit_path, 'X_ray%s_1/ImageSets/Main/%s.txt' % (year, image_set)),
                             encoding='utf-8').read().strip().split()
            list_file_1 = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file_1.write('%s/X_ray%s_1/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation_1(year, image_id, list_file_1)
                list_file_1.write('\n')
            list_file.close()
        print("Generate 2022_train.txt and 2022_val.txt for train done.")
