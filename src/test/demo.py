import argparse
import cv2
import os 
import sys
import numpy as np
import torch
from minist_test import MinistTest
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from Alexnet import AlexNet
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from load_imgs import ReadDataset
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from char_seg import Cardseg

def test(im_dir):
    charReg = MinistTest('lrr_mnist_cnn_50.pt')
    dir_cnts = os.listdir(im_dir)
    tpr=0
    total = 0
    # charseg = Cardseg(im,'./output')
    # if len(charseg)>0:
    #     char_result=charReg.inference(charseg[:3])
    #     reg_out = char_result.argmax(dim=1).numpy()
    #     print(reg_out)
    for tmp in dir_cnts:
        imgpath = os.path.join(im_dir,tmp.strip())
        im = cv2.imread(imgpath)
        if im is None:
            continue
        charseg = Cardseg(im,'./output/')
        if len(charseg)>0:
            total+=1
            char_result = charReg.inference(charseg[:3])
            #print(char_result)
            reg_out = char_result.argmax(dim=1).numpy()
            reg_out = list(map(str,reg_out))
            reg_out = ''.join(reg_out)
            if reg_out == tmp[:-4]:
                tpr+=1
            print('real: %s \t pred: %s' % (tmp[:-4],reg_out))
    print('total: %d \t right num: %d ',(total,tpr))