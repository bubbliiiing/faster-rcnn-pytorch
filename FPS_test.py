import colorsys
import copy
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm

from frcnn import FRCNN
from nets.frcnn import FasterRCNN
from utils.utils import DecodeBox, get_new_img_size, loc2bbox, nms

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，而且处理过程包含了前处理和绘图部分。
'''
class FPS_FRCNN(FRCNN):
    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        old_width, old_height = image_shape[1], image_shape[0]
        old_image = copy.deepcopy(image)
        
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        width,height = get_new_img_size(old_width, old_height)
        image = image.resize([width,height], Image.BICUBIC)

        #-----------------------------------------------------------#
        #   图片预处理，归一化。
        #-----------------------------------------------------------#
        photo = np.transpose(np.array(image,dtype = np.float32)/255, (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.model(images)
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = self.iou, score_thresh = self.confidence)
            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            if len(outputs)>0:
                outputs = np.array(outputs)
                bbox = outputs[:,:4]
                label = outputs[:, 4]
                conf = outputs[:, 5]

                bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
                bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                roi_cls_locs, roi_scores, rois, _ = self.model(images)
                #-------------------------------------------------------------#
                #   利用classifier的预测结果对建议框进行解码，获得预测框
                #-------------------------------------------------------------#
                outputs = self.decodebox.forward(roi_cls_locs[0], roi_scores[0], rois, height = height, width = width, nms_iou = self.iou, score_thresh = self.confidence)
                #---------------------------------------------------------#
                #   如果没有检测出物体，返回原图
                #---------------------------------------------------------#
                if len(outputs)>0:
                    outputs = np.array(outputs)
                    bbox = outputs[:,:4]
                    label = outputs[:, 4]
                    conf = outputs[:, 5]

                    bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
                    bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
        
frcnn = FPS_FRCNN()
test_interval = 100
img = Image.open('img/street.jpg')
tact_time = frcnn.get_FPS(img, test_interval)
print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
