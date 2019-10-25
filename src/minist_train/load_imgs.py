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
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h,w = img.shape[:2]
        if h !=28 or w !=28:
            img = cv2.resize(img,(28,28))
        img = 255-img
        if len(img.shape)==2:
            img = np.expand_dims(img,2)
        img = img.astype(np.float32)/255.0
        gt_box_label = np.array(tmp_annotation[1])
        return img,gt_box_label
    
    def num_classes(self):
        return 10


def generate_list_from_dir(dirpath,out_file):
    '''
    dirpath: saved images path
            "dirpath/id_num/image1.jpg"
    return: images paths txtfile
            "id_num/img1.jpg"
    '''
    f_w = open('train.txt','w')
    f2_w = open('test.txt','w')
    #name_pick = out_file.split('.')[0]
    #name_wr = open("../output/yunli_id2name.txt",'w')
    files = os.listdir(dirpath)
    total_ = len(files)
    print("total id ",len(files))
    idx =0
    file_name = []
    total_cnt = 0
    test_cnt = 0
    for file_cnt in files:
        img_dir = os.path.join(dirpath,file_cnt)
        imgs = os.listdir(img_dir)
        idx+=1
        sys.stdout.write("\r>>convert  %d/%d" %(idx,total_))
        sys.stdout.flush()
        test_cnt = 0
        tmp_cnt = len(imgs)
        for img_one in imgs:
            img_path = os.path.join(file_cnt,img_one)
            total_cnt+=1
            if test_cnt <= 200:
                f_w.write("{},{}\n".format(img_path,file_cnt))
            elif test_cnt < 300:
                f2_w.write("{},{}\n".format(img_path,file_cnt))
            else:
                break
            test_cnt+=1
        #name_list = file_cnt.split()
        #name_wr.write("{},{}\n".format(name_list[0],name_list[1]))
    print("total img ",total_cnt)
    f_w.close()
    f2_w.close()
    #name_wr.close()

if __name__=='__main__':
    generate_list_from_dir('/data/detect/num_pic','./')