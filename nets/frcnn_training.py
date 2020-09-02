
import random
import numpy as np
from random import shuffle
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_new_img_size(width, height, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height
    
class Generator(object):
    def __init__(self,train_lines,shape=[600,600],batch_size=1):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.shape = shape
        
    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h,w = self.shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(.5, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    
    def generate(self):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            b = 0
            imgs = []
            boxes = []
            labels = []
            for annotation_line in lines:  
                img,y=self.get_random_data(annotation_line)
                
                if len(y)==0:
                    continue
                box = np.array(y[:,:4],dtype=np.float32)
                box[:,0] = y[:,0]
                box[:,1] = y[:,1]
                box[:,2] = y[:,2]
                box[:,3] = y[:,3]
                
                box_widths = box[:,2] - box[:,0]
                box_heights = box[:,3] - box[:,1]
                if (box_heights<=0).any() or (box_widths<=0).any():
                    continue
                label = y[:,-1]
                img = img / 255.0

                imgs.append(np.transpose(img,[2,0,1]))
                boxes.append(box)
                labels.append(label)
                b += 1
                if self.batch_size == b:
                    imgs = np.array(imgs)
                    boxes = np.array(boxes)
                    labels = np.array(labels)
                    b = 0
                    yield imgs,boxes,labels
                    imgs = []
                    boxes = []
                    labels = []
                            
