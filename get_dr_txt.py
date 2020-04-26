#----------------------------------------------------#
#   获取测试集的detection-result和images-optional
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
from frcnn import FRCNN
from PIL import Image
from torch.autograd import Variable
import torch
import numpy as np
import os
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from utils.utils import loc2bbox, nms, DecodeBox
from nets.frcnn import FasterRCNN
from nets.frcnn_training import get_new_img_size
from PIL import Image, ImageFont, ImageDraw
import copy

MEANS = (104, 117, 123)
class mAP_FRCNN(FRCNN):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.05
        f = open("./input/detection-results/"+image_id+".txt","w") 
        image_shape = np.array(np.shape(image)[0:2])
        old_width = image_shape[1]
        old_height = image_shape[0]
        width,height = get_new_img_size(old_width,old_height)
        image = image.resize([width,height])
        photo = np.array(image,dtype = np.float32)/255
        photo = np.transpose(photo, (2, 0, 1))
        
        images = []
        images.append(photo)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()

        roi_cls_locs, roi_scores, rois, roi_indices = self.model(images)
        decodebox = DecodeBox(self.std, self.mean, self.num_classes)
        outputs = decodebox.forward(roi_cls_locs, roi_scores, rois, height=height, width=width, score_thresh = self.confidence)
        if len(outputs)==0:
            return
        bbox = outputs[:,:4]
        conf = outputs[:, 4]
        label = outputs[:, 5]

        bbox[:, 0::2] = (bbox[:, 0::2])/width*old_width
        bbox[:, 1::2] = (bbox[:, 1::2])/height*old_height
        bbox = np.array(bbox,np.int32)
        for i, c in enumerate(label):
            predicted_class = self.class_names[int(c)]
            score = conf[i]

            left, top, right, bottom = bbox[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

frcnn = mAP_FRCNN()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")


for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    image.save("./input/images-optional/"+image_id+".jpg")
    frcnn.detect_image(image_id,image)
    print(image_id," done!")
    

print("Conversion completed!")