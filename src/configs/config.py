import easydict

cfgs = easydict.EasyDict()

#********************************* Character segmentation configs
cfgs.SZ = 56          #img output size
cfgs.MAX_WIDTH = 1000 #the input img width limit
cfgs.Min_Area = 2000  #the area of object
cfgs.PROVINCE_START = 1000
cfgs.Min_char_WIDTH = 4 # the min width of a character
#****************************** network training configs
cfgs.DataDir = '/wdc/LXY.data/num_pic_train'
cfgs.TestDir = '/wdc/LXY.data/num_pic'
cfgs.ModelPrefix = 'lee_mnist_cnn'
cfgs.ModelSaveDir = '/wdc/LXY.data/models/pytorch_minist'
cfgs.LogInterval = 10
cfgs.ModelSaveInterval = 10
cfgs.LogDir = '../logs'