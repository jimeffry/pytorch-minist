import sys
import os
import cv2
import torch
import numpy as np

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
            if test_cnt <= 20000:
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

def copyimg(infile,orgdir,disdir):
    fin = open(infile)
    fcnts = fin.readlines()
    if not os.path.exists(disdir):
        os.makedirs(disdir)
    for tmpa in fcnts:
        tmp = tmpa.strip().split(',')[0]
        ts = tmp.split('/')
        dispath = os.path.join(disdir,ts[0])
        orgpath = os.path.join(orgdir,tmp)
        img = cv2.imread(orgpath)
        img = 255-img
        if not os.path.exists(dispath):
            os.makedirs(dispath)
        spath = os.path.join(dispath,ts[-1])
        cv2.imwrite(spath,img)

if __name__=='__main__':
    #generate_list_from_dir('/data/detect/num_pic','./')
    copyimg('train.txt','/data/detect/num_pic','/data/detect/num_pic_train')