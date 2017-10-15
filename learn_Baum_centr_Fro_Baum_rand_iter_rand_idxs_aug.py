import torch
import torch.nn as nn
import numpy as np
from  scipy.ndimage import zoom as imzoom
import sys
import os
import math
from PIL import Image
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
from pytorch_sift import SIFTNet
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm

USE_CUDA = True

LOG_DIR = 'log_snaps'
BASE_LR = 0.001
start = 0
end = 25
n_epochs = end - start
from SpatialTransformer2D import SpatialTransformer2d
from HardNet import HardNet
#hardnet = HardNet()
#checkpoint = torch.load('HardNetLib.pth')
#hardnet.load_state_dict(checkpoint['state_dict'])
from HandCraftedModules import AffineShapeEstimator

from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs
from ReprojectonStuff import get_GT_correspondence_indexes_Fro,get_GT_correspondence_indexes,get_GT_correspondence_indexes_Fro_and_center, affineAug
class BaumNet(nn.Module):
    def __init__(self, PS = 16):
        super(BaumNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = True),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = True),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = True),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = True),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1, bias = True),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 3, kernel_size=4, bias = True),
            nn.Tanh()
        )
        self.PS = PS
        self.features.apply(weights_init)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        abc = self.features(self.input_norm(input))
        a = (abc[:,0,:,:].contiguous()  + 1.0)
        b = (abc[:,1,:,:].contiguous()  + 0.0)
        c = (abc[:,2,:,:].contiguous()  + 1.0)
        return abc2A(a,b,c) , 1.0

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        try:
            nn.init.constant(m.bias.data, 0.0)
        except:
            pass
    return
    
def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    n_triplets = 116*5.
    n_epochs = 10.
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] =  BASE_LR *  (1.0 - float(group['step']) * float(1.0) / (n_triplets * float(n_epochs)))
    return

def create_optimizer(model, new_lr, wd):
    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                          momentum=0.9, dampening=0.5,
                          weight_decay=wd)
    return optimizer

def create_loaders(load_random_triplets = False):

    kwargs = {'num_workers': 2, 'pin_memory': True} if True else {}

    transform = transforms.Compose([
            transforms.ToTensor()])
    #        transforms.Normalize((args.mean_image,), (args.std_image,))])

    train_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/storage/learned_detector/dataset/', 'b',
                             train=True, transform=None,
                             download=True), batch_size = 1,
        shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/storage/learned_detector/dataset/', 'b',
                             train=False, transform=None,
                             download=True), batch_size = 1,
        shuffle = False, **kwargs)

    return train_loader, test_loader

def train(train_loader, model, optimizer, epoch, cuda = True):
    # switch to train mode
    model.train()
    log_interval = 1
    total_loss = 0
    total_feats = 0
    spatial_only = True
    pbar = enumerate(train_loader)
    for batch_idx, data in pbar:
        print 'Batch idx', batch_idx
        #print model.detector.shift_net[0].weight.data.cpu().numpy()
        img1, img2, H1to2  = data
        #if np.abs(np.sum(H.numpy()) - 3.0) > 0.01:
        #    continue
        H1to2 = H1to2.squeeze(0)
        if (img1.size(3) *img1.size(4)   > 1340*1000):
            print img1.shape, ' too big, skipping'
            continue
        img1 = img1.float().squeeze(0)
        img2 = img2.float().squeeze(0)
        if cuda:
            img1, img2, H1to2 = img1.cuda(), img2.cuda(), H1to2.cuda()
        img1, img2, H1to2 = Variable(img1, requires_grad = False), Variable(img2, requires_grad = False), Variable(H1to2, requires_grad = False)
        new_img2, H_Orig2New = affineAug(img2)
        H1to2new = torch.mm(H_Orig2New, H1to2)
        #print H1to2
        LAFs1, aff_norm_patches1, resp1 = HA(img1, True, True, True)
        LAFs2, aff_norm_patches2, resp2 = HA(new_img2, True, True)
        if (len(LAFs1) == 0) or (len(LAFs2) == 0):
            optimizer.zero_grad()
            continue
        fro_dists, idxs_in1, idxs_in2, LAFs2_in_1 = get_GT_correspondence_indexes_Fro_and_center(LAFs1,LAFs2, H1to2new,  dist_threshold = 4., 
                                                                             center_dist_th = 7.0,
                                                                            skip_center_in_Fro = True,
                                                                            do_up_is_up = True,return_LAF2_in_1 = True);
        if  len(fro_dists.size()) == 0:
            optimizer.zero_grad()
            print 'skip'
            continue
        aff_patches_from_LAFs2_in_1 = extract_patches(img1, normalizeLAFs(LAFs2_in_1[idxs_in2.data.long(),:,:], img1.size(3), img1.size(2)))
         
        #loss = fro_dists.mean()
        patch_dist = torch.sqrt((aff_norm_patches1[idxs_in1.data.long(),:,:,:]/100. - aff_patches_from_LAFs2_in_1/100.) **2 + 1e-8).view(fro_dists.size(0),-1).mean(dim = 1)
        loss = (fro_dists * patch_dist).mean()
        print 'Fro dist', fro_dists.mean().data.cpu().numpy()[0], loss.data.cpu().numpy()[0]
        total_loss += loss.data.cpu().numpy()[0]
        #loss += patch_dist
        total_feats += fro_dists.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #adjust_learning_rate(optimizer)
        print epoch,batch_idx, loss.data.cpu().numpy()[0], idxs_in1.shape

    print 'Train total loss:', total_loss / float(batch_idx+1), ' features ', float(total_feats) / float(batch_idx+1)
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/elu_new_checkpoint_{}.pth'.format(LOG_DIR, epoch))

def test(test_loader, model, cuda = True):
    # switch to train mode
    model_num_feats = model.num
    model.num = 1500;
    model.eval()
    log_interval = 1
    pbar = enumerate(test_loader)
    total_loss = 0
    total_feats = 0
    for batch_idx, data in pbar:
        print 'Batch idx', batch_idx
        img1, img2, H1to2  = data
        if (img1.size(3) *img1.size(4)   > 1500*1200):
            print img1.shape, ' too big, skipping'
            continue
        H1to2 = H1to2.squeeze(0)
        img1 = img1.float().squeeze(0)
        img2 = img2.float().squeeze(0)
        if cuda:
            img1, img2, H1to2 = img1.cuda(), img2.cuda(), H1to2.cuda()
        img1, img2, H1to2 = Variable(img1, volatile = True), Variable(img2, volatile = True), Variable(H1to2, volatile = True)
        LAFs1, aff_norm_patches1, resp1 = HA(img1)
        LAFs2, aff_norm_patches2, resp2 = HA(img2)
        if (len(LAFs1) == 0) or (len(LAFs2) == 0):
            continue
        fro_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes_Fro_and_center(LAFs1,LAFs2, H1to2, 
                                                                                     dist_threshold = 3.,
                                                                             center_dist_th = 7.0,
                                                                            skip_center_in_Fro = True,
                                                                            do_up_is_up = True);
        if  len(fro_dists.size()) == 0:
            print 'skip'
            continue
        loss = fro_dists.mean()
        total_feats += fro_dists.size(0)
        total_loss += loss.data.cpu().numpy()[0]
        print 'test img', batch_idx, loss.data.cpu().numpy()[0], fro_dists.size(0)
    print 'Total loss:', total_loss / float(batch_idx+1), 'features', float(total_feats) / float(batch_idx+1)
    model.num = model_num_feats

train_loader, test_loader = create_loaders()

HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 350, border = 5, num_Baum_iters = 2, AffNet = BaumNet())


model = HA
if USE_CUDA:
    model = model.cuda()

optimizer1 = create_optimizer(model.AffNet.features, BASE_LR, 1e-4)

#test(test_loader, model, cuda = USE_CUDA)
for epoch in range(n_epochs):
    print 'epoch', epoch
    if USE_CUDA:
        model = model.cuda()
    train(train_loader, model, optimizer1, epoch, cuda = USE_CUDA)
    test(test_loader, model, cuda = USE_CUDA)
