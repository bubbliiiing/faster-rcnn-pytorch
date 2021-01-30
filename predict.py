'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        #-------------------------------------#
        #   转换成RGB图片，可以用于灰度图预测。
        #-------------------------------------#
        image = image.convert("RGB")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        r_image.show()
