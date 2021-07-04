import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import  time #tcw20182159tcw
from torch.autograd import Variable
from models import DudeNet
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(description="DudeNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='cc-noisy', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=15, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    aa = 'save_image/'
    if not os.path.exists(aa):
        os.mkdir(aa)
    save_dir = aa + 'sigma' + str(opt.test_noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
        os.mkdir(save_dir)
    net = DudeNet(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_1.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        #print Img.shape
        Img = torch.tensor(Img)
        #print Img.shape
        Img = Img.permute(2,0,1)
        Img = Img.numpy()
        a1, a2, a3 = Img.shape
        #print a1, a2,a3
        Img = np.tile(Img,(3,1,1,1))  #expand the dimensional 
        #print Img.shape
        Img = np.float32(normalize(Img))
        #print Img.shape
        ISource = torch.Tensor(Img)
        # noise
        #ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())#tcw
        #ISource = ISource.cuda()
        #INoisy = INoisy.cuda()
        ISource = Variable(ISource) #tcw201809131503tc
        ISource= ISource.cuda() #tcw201809131503
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(ISource), 0., 1.)
            out1 = Out[0,:,:,:]
            out1 = out1.permute(1,2,0)
            out1 = out1.cpu().numpy() 
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
        a = os.path.basename(f)
        cv2.imwrite(os.path.join(save_dir,a),out1*255.0)
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
