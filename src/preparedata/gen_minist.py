# -*- coding:utf-8 -*-

from PIL import Image,ImageFont,ImageDraw,ImageFilter
import random
import os
import time
import tqdm

class Captcha(object):
    def __init__(self,size=(20,24),fontSize=20):
        self.font = ImageFont.truetype('./fonts/Arial.ttf',fontSize)
        self.size = size
        self.image = Image.new('RGBA',self.size,(255,)*4)
        self.text = ''

    def rotate(self, angle):
        rot = self.image.rotate(angle,expand=0)
        fff = Image.new('RGBA',rot.size,(255,)*4)
        self.image = Image.composite(rot,fff,rot)
        #self.image = fff-self.image
        #self.image = Image.Image.convert("RGB", self.image)

    def randColor(self):
        self.fontColor = (255,255,255)


    def setNum(self, num):
        return num

    def write(self,text,x,y):
        draw = ImageDraw.Draw(self.image)
        draw.text((x,y),text,fill=self.fontColor,font=self.font)

    def writeNum(self, num, angle,x=2,y=-2):
        # x = 2
        # y = -2
        self.text = num
        self.fontColor = (0, 0, 0)
        self.write(num, x, y)
        self.rotate(angle)
        return self.text

    def save(self, save_path):
        # self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
        self.image.save(save_path)

pic_root_path = '/data/detect/num_pic'
if not os.path.exists(pic_root_path):
    os.mkdir(pic_root_path)

angles = [(-30,30)]
for i in tqdm.tqdm(range(10)):
    pic_num_path = os.path.join(pic_root_path, str(i))
    if not os.path.exists(pic_num_path):
        os.mkdir(pic_num_path)
    for angle_i in angles:
        angle_name = str(angle_i[0])+'_'+str(angle_i[1])
        #pic_angle_path = os.path.join(pic_num_path, angle_name)
        # if not os.path.exists(pic_angle_path):
        #     os.mkdir(pic_angle_path)
        x_oders = [5,8,10,20,30]
        y_oders = [-5,-10,-20,-30,-25]
        imgsizes = [28,56,112,120,130]
        for idx,fontsize in enumerate([28,56,112,150,200]):
            tmpsz = imgsizes[idx]
            x = x_oders[idx]
            y = y_oders[idx]
            for j in range(1000):
                # Keep 5 decimal places
                angle = round(random.uniform(angle_i[0], angle_i[1]),5) 
                img = Captcha(size=(tmpsz, tmpsz), fontSize=fontsize)
                num = img.writeNum(str(i), angle,x,y)

                img_name = str(j)+'_'+str(fontsize)+'_'+str(angle)+'_'+str(num)+'.png'

                save_path = os.path.join(pic_num_path, img_name)
                img.save(save_path)