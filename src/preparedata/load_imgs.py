import sys
import os
import cv2
import torch
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ReadDataset(Dataset):
    """
    VOC coco Detection Dataset Object
    """
    def __init__(self,voc_file,voc_dir,transf=None):
        self.voc_file = voc_file
        self.voc_dir = voc_dir
        self.ids = []
        self.annotations = []
        self.labels = {}
        self.load_txt()
        self.idx = 0
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.transform = transf
        

    def __getitem__(self, index):
        img,annot = self.pull_item(index)
        if self.transform:
            img = self.transform(img)
            
        return img,annot

    def __len__(self):
        return len(self.annotations)

    def load_txt(self):
        self.voc_r = open(self.voc_file,'r')
        voc_annotations = self.voc_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.voc_dir,tmp_splits[0])
            label = float(tmp_splits[1])
            bbox = [img_path,label]
            self.annotations.append(bbox)
        
        
    def close_txt(self):
        self.voc_r.close()
        
    
    def pull_item(self, index):
        '''
        output: img - shape(c,h,w)
                gt_boxes+label: box-(x1,y1,x2,y2)
                label: dataset_class_num 
        
        if self.idx >= self.total_num -1:
            random.shuffle(self.shulf_num)
            self.idx = 0
        for tmp_idx in range(batch_size):
            if self.idx >= self.total_num:
                rd_idx = 0
            else:
                rd_idx = self.shulf_num[self.idx]
        '''
        tmp_annotation = self.annotations[index]
        img_path = tmp_annotation[0]
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        if h !=28 or w !=28:
            img = cv2.resize(img,(28,28))
        #img = 255-img
        if len(img.shape)==3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)/255.0
        gt_box_label = np.array(tmp_annotation[1])
        return torch.from_numpy(img).unsqueeze(0),gt_box_label
    
    def num_classes(self):
        return 10




