from __future__ import print_function
import argparse
import cv2
import os 
import sys
import numpy as np
import time
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__),'../network'))
from Alexnet import AlexNet
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from load_imgs import ReadDataset
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def params():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train_file', type=str, default='train.txt',
                        help='input train file')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test_file', type=str, default='test.txt', 
                        help='input test file')
    parser.add_argument('--load_num', type=int, default=None,
                        help='fintuing num')
    parser.add_argument('--data_dir', type=str, default='/wdc/LXY.data/num_pic',
                        help='For Saving the current Model')
    return parser.parse_args()
    
def train(args, model, device, train_loader, optimizer, epoch,logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # cv2.imshow('src',np.array(data.cpu().numpy()[0,0]*255,dtype=np.uint8))
        # cv2.waitKey(0)
        optimizer.zero_grad()
        output = model(data)
        #print(output.argmax(dim=1))
        target = target.long()
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % cfgs.LogInterval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Lr: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),optimizer.param_groups[0]['lr']))

def test(args, model, device, test_loader,logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.long()
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def main():
    # Training settings
    args = params()
    data_dir = cfgs.DataDir
    test_dir = cfgs.TestDir
    train_file = args.train_file
    test_file = args.test_file
    model_dir = cfgs.ModelSaveDir
    model_path= os.path.join(model_dir,cfgs.ModelPrefix)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #*******************************************************************************create logg
    log_dir = cfgs.LogDir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    log_name = time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    log_path = os.path.join(log_dir,log_name)
    hdlr = logging.FileHandler(log_path)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    #*******************************************************************************load data
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    #     #transforms.Normalize((0.1307,), (0.3081,))
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    #***************
    
    dataset_train = ReadDataset(train_file,data_dir,transf=transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])) #transforms.Normalize((0.1307,), (0.3081,))
    train_loader = DataLoader(dataset_train, args.batch_size,
                                  num_workers=1,
                                  shuffle=True, 
                                  pin_memory=True)
    dataset_test = ReadDataset(test_file,test_dir,transf=transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))]))
    test_loader = DataLoader(dataset_test, args.batch_size,
                                  num_workers=1,
                                  shuffle=True, 
                                  pin_memory=True)
    # load model
    model = AlexNet()
    if args.load_num is not None:
        loadpath = model_path +'_'+str(args.load_num)+'.pt'
        state_dict = torch.load(loadpath,map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,logger)
        ave_loss = test(args, model, device, test_loader,logger)
        if epoch % cfgs.ModelSaveInterval ==0:
            torch.save(model.state_dict(),model_path+"_%d.pt" % epoch)
        # scheduler.step(ave_loss)

    if (args.save_model):
        torch.save(model.state_dict(),model_path+".pt")
        
if __name__ == '__main__':
    main()