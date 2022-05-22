import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        # "model_path"        : 'logs/ep200-loss0.682-val_loss0.826.pth',
        "model_path": 'logs/your_pth',

        "classes_path"      : 'model_data/voc_classes.txt',

        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],

        "input_shape"       : [416, 416],

        "confidence"        : 0.5,
        # "confidence": 0.2,

        "nms_iou"           : 0.3,

        "letterbox_image"   : False,

        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)


        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()


    def generate(self):

        self.net    = YoloBody(self.anchors_mask, self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')##########
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()


    def detect_image(self, image,image_1):

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)
        image_1 = cvtColor(image_1)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data_1 = resize_image(image_1, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_data_1 = np.expand_dims(np.transpose(preprocess_input(np.array(image_data_1, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_1 = torch.from_numpy(image_data_1)
            if self.cuda:
                images = images.cuda()
                images_1 = images_1.cuda()

            outputs = self.net(images,images_1)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * image.size[1] + 5).astype('int32')+10)
        # thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape)+10, 1))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            # print(top)
            left    = max(0, np.floor(left).astype('int32'))
            # print(left)
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            # print(bottom)
            right   = min(image.size[0], np.floor(right).astype('int32'))
            # print(right)

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # draw.rectangle([left + i-10, top + i-10, right - i+10, bottom - i+10], outline=self.colors[c])
                draw.rectangle([left + i , top + i , right - i , bottom - i ], outline=self.colors[c])
            # draw.rectangle([tuple(text_origin-10), tuple(text_origin + label_size)], fill=self.colors[c])
            # draw.text(text_origin-10, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)

                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, image_1_id, image_1,class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)
        image_1     = cvtColor(image_1)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_1_data= resize_image(image_1, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image_1_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_1_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images_1 = torch.from_numpy(image_1_data)
            if self.cuda:
                images = images.cuda()
                images_1 = images_1.cuda()

            outputs = self.net(images,images_1)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
