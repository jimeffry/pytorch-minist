import sys
import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from Alexnet import AlexNet

class MinistTest(object):
    def __init__(self, model_path, use_cuda=True):
        #self.net = Net()
        self.net = AlexNet()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path,map_location=self.device)
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (28, 28)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        

    def preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)
        # for im in im_crops:
        #     print(im.shape)
        #     print(self.norm(_resize(im, self.size)).size())
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def inference(self, im_crops):
        im_batch = self.preprocess(im_crops)
        #print(im_batch.size())
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu()


if __name__ == '__main__':
    img = cv2.imread("/data/detect/Archive/t1.png")[:,:,0]
    imgin = [img,img]
    extr = MinistTest("/data/gen_mnist_cnn.pt")
    feature = extr.inference(imgin)
    print(feature.shape)

