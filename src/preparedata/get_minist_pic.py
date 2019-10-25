import cv2
import os 
import sys
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader


if __name__=='__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=8, shuffle=True, **kwargs)
    num_dict = dict()
    save_dir = '/data/detect/minist'
    for batch_idx, (data, target) in enumerate(train_loader):
        tmp_data = data.cpu().numpy()
        tmp_lab = target.cpu().numpy()
        for i in range(tmp_data.shape[0]):
            tmp = np.array(tmp_data[i]*255,dtype=np.uint8)
            tmp = tmp.transpose((1,2,0))
            tmp_key = str(int(tmp_lab[i]))
            tmp_cnt = num_dict.setdefault(tmp_key,0)
            tmp_dir = os.path.join(save_dir,tmp_key)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            tmp_path = os.path.join(tmp_dir,'num'+'_'+str(tmp_cnt)+'.jpg')
            cv2.imwrite(tmp_path,tmp)
            num_dict[tmp_key] = tmp_cnt+1
        if batch_idx % 100:
            print(batch_idx)