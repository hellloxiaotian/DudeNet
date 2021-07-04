import cv2
import os
import  glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import  time #tcw20182159tcw
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss #TCW20180913TCW
from models import DudeNet
from dataset_r import ImageDataset #tcw201812041630
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DudeNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
parser.add_argument("--test_data",type=str,default='cc',help='test on cc')
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def normalize(data):
   return data/255.

def main():
    # Load dataset
    t1 = time.clock()
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    aa = "data"
    train_loader = DataLoader(ImageDataset(aa, noisy_image= "fadnettrain_noisyimages",clean_image="fadnettrain_labelimages"),batch_size =opt.batchSize,shuffle=True,num_workers=4) 
    net = DudeNet(channels=3, num_of_layers=opt.num_of_layers)
    criterion = nn.MSELoss(size_average=False)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda() #tcw201810192202tcw 
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    psnr_list = [] #201809062254tcw. it is used to save the psnr value of each epoch
    patch_num = len(train_loader)
    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > opt.milestone and  epoch <=60:
            current_lr  =  opt.lr/10. 
        if epoch > 60  and  epoch <=90:
            current_lr = opt.lr/100.
        if epoch > 90:
            current_lr = opt.lr/1000.
        for param_group in optimizer.param_groups:
	    param_group["lr"] = current_lr
	    print('learning rate %f' % current_lr)
        for i, data in enumerate(train_loader):
	    model.train()
	    noisy = data['input'].cuda()
	    clean = data['target'].cuda()
	    noisy = Variable(noisy)
	    clean = Variable(clean)
	    our_predicted_cleanimage = model(noisy)
	    loss =  criterion(our_predicted_cleanimage, clean) / (noisy.size()[0]*2)
	    optimizer.zero_grad() #tcw201809112015tcw
	    loss.backward()
	    optimizer.step()
	    model.eval()
	    our_predicted_cleanimage1 = torch.clamp(model(noisy), 0., 1.) 
	    psnr_train = batch_PSNR(our_predicted_cleanimage1, clean, 1.)
	    print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1,patch_num, loss.item(), psnr_train))
        model.eval() #tcw20180915tcw 
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' #tcw201809071117tcw
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) #tcw201809062210tcw
        psnr_val = 0
        files_source = glob.glob(os.path.join('data', opt.test_data, '*_real.png'))
        files_source.sort()
        psnr_test = 0
        for f in files_source:
            Img_test = cv2.imread(f)
            Img_test = torch.tensor(Img_test)
            Img_test = Img_test.permute(2,0,1)
            Img_test = Img_test.numpy()
            a1, a2, a3 = Img_test.shape
            Img_test = np.tile(Img_test,(3,1,1,1)) 
            Img_test = np.float32(normalize(Img_test))
            ISource = torch.Tensor(Img_test)
            ISource = ISource.cuda()
            Img_test_clean = f[:-9] + '_mean.png'
            Img_test_clean = cv2.imread(Img_test_clean)
            Img_test_clean = torch.tensor(Img_test_clean)
            Img_test_clean = Img_test_clean.permute(2,0,1)
            Img_test_clean = Img_test_clean.numpy()
            a1, a2, a3 = Img_test_clean.shape
            Img_test_clean = np.tile(Img_test_clean,(3,1,1,1)) 
            Img_test_clean = np.float32(normalize(Img_test_clean))
            Img_test_clean = torch.Tensor(Img_test_clean)
            Img_test_clean  = Img_test_clean .cuda()
            with torch.no_grad(): # this can save much memory
                Out  = torch.clamp(model(ISource), 0., 1.)
            psnr = batch_PSNR(Out, Img_test_clean, 1.)
            psnr_list.append(psnr)
            psnr_test += psnr
            print("%s PSNR %f" % (f, psnr))
        psnr_test /= len(files_source)
        print("\nPSNR on test data %f" % psnr_test)
    filename = save_dir + 'psnr.txt' #tcw201809071117tcw
    f = open(filename,'w') #201809071117tcw
    for line in psnr_list:  #201809071117tcw
        f.write(line+'\n') #2018090711117tcw
    f.close()
    t2 = time.clock()
    print t2-t1
if __name__ == "__main__":
    if opt.preprocess:
        main()
