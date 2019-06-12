# coding=utf-8
"""-----------------------------------------
采集我的人脸数据集
-----------------------------------------"""
import cv2
import dlib
import os
import random

faces_my_path = './faces_my'
size = 64
if not os.path.exists(faces_my_path):
    os.makedirs(faces_my_path)

"""改变图片的亮度与对比度"""
def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,k] = tmp
    return img

"""特征提取器"""
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

num = 1
while True:
    if (num <= 10000):
        print('Being processed picture %s' % num)
        success, img = cap.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """使用特征提取器进行人脸检测"""
        dets = detector(gray_img, 1)

        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1,x2:y2]
            """调整对比度与亮度， 都取随机数"""
            face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            face = cv2.resize(face, (size,size))
            cv2.imshow('image', face)
            cv2.imwrite(faces_my_path+'/'+str(num)+'.jpg', face)
            num += 1
        key = cv2.waitKey(30)
        if key == 27:
            break
    else:
        print('Finished!')
        break
