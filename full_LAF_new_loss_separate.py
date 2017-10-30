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
BASE_LR = 0.0005
start = 0
end = 1000
n_epochs = end - start
from HandCraftedModules import AffineShapeEstimator

from SparseImgRepresenterLAF import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A, extract_patches,normalizeLAFs
from ReprojectonStuff import get_GT_correspondence_indexes_Fro,get_GT_correspondence_indexes,get_GT_correspondence_indexes_Fro_and_center, affineAug
from ReprojectonStuff import reprojectLAFs, distance_matrix_vector,ratio_matrix_vector,LAFs_to_H_frames
from ReprojectonStuff import LAFMagic, pr_l,identity_loss
class LAFNet(nn.Module):
    def __init__(self, PS = 16):
        super(LAFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.psi_net =  nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=4, bias = True),
        )
        self.theta_net = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=4, bias = True),
        )
        self.shift_net = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=4, bias = True),
            nn.Tanh()
        )
        self.iso_scale_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=4, bias = True),
            nn.Tanh()
        )
        self.horizontal_tilt_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=4, bias = True),
            nn.Sigmoid()
        )
        
        self.PS = PS
        self.features.apply(weights_init)
        self.eye2 = torch.autograd.Variable(torch.eye(2))
        self.zero = torch.autograd.Variable(torch.zeros(1,1,1))
        self.one = torch.autograd.Variable(torch.ones(1,1,1))
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.view(-1, 1, 1);
        sin_a = torch.sin(angle_in_radians)
        cos_a = torch.cos(angle_in_radians)
        A1_x = torch.cat([cos_a, sin_a], dim = 2)
        A2_x = torch.cat([-sin_a, cos_a], dim = 2)
        transform = torch.cat([A1_x,A2_x], dim = 1)
        return transform

    def get_tilt_matrix(self, horizontal_tilt): #X-axis anisoptropic scale
        an_s = horizontal_tilt.view(-1, 1, 1)
        A1_x = torch.cat([an_s, self.zero.expand_as(an_s)], dim = 2)
        A2_x = torch.cat([self.zero.expand_as(an_s), self.one.expand_as(an_s) / an_s], dim = 2)
        return torch.cat([A1_x,A2_x], dim = 1)
    
    def get_scale_matrix(self, iso_scale): #Isotropic scale
        iso_s = iso_scale.view(-1, 1, 1)
        A1_x = torch.cat([iso_s, self.zero.expand_as(iso_s)], dim = 2)
        A2_x = torch.cat([self.zero.expand_as(iso_s), iso_s], dim = 2)
        return torch.cat([A1_x,A2_x], dim = 1)
    
    def compose_affine_matrix(self, psi, theta, iso_scale, horizontal_tilt, shift): 
        ## See illustration in MODS paper for notation
        # Output is n x 2 x 3 Affine transformation matrix 
        in_plane_rot =  self.get_rotation_matrix(psi);
        out_plane_rot = self.get_rotation_matrix(theta);
        tilt_matrix = self.get_tilt_matrix(horizontal_tilt);
        iso_scale = iso_scale.view(-1,1,1)
        A_iso_scale  = self.get_scale_matrix(iso_scale);
        A_iso_scale_in_plane = torch.bmm(A_iso_scale,in_plane_rot);
        A_tilt_out_of_place = torch.bmm(tilt_matrix, out_plane_rot)
        A_no_shift = torch.bmm(A_iso_scale_in_plane, A_tilt_out_of_place)
        return torch.cat([A_no_shift, shift.view(-1,2,1)], dim = 2)

    def forward(self, input):
        if input.is_cuda:
            self.eye2 = self.eye2.cuda()
            self.zero = self.zero.cuda()
            self.one = self.one.cuda()
        else:
            self.eye2 = self.eye2.cpu()
            self.zero = self.zero.cpu()
            self.one = self.one.cpu()
        ST_features = self.features(self.input_norm(input))
        
        psi_c_s = self.psi_net(ST_features).view(-1,2) # change to sin cos
        theta_c_s = self.theta_net(ST_features).view(-1,2) # change to sin cos
        psi = torch.atan2(psi_c_s[:,1],psi_c_s[:,0] + 1e-12)
        theta = torch.atan2(theta_c_s[:,1],theta_c_s[:,0] + 1e-12)
        
        shift = 0.5 * self.shift_net(ST_features) 
        tilt = torch.clamp(0.5 +  self.horizontal_tilt_net(ST_features),
                          min = 0.5, max = 2.0)
        scale = torch.clamp(1.0 + 0.5* self.iso_scale_net(ST_features),
                            min = 0.3, max = 2.0)
        out = self.compose_affine_matrix(psi, theta, scale, tilt, shift)
        return out, scale, (psi,theta,scale,tilt,shift) 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal(m.weight.data)
        m.weight.data = 0.1 * m.weight.data;
        try:
            nn.init.constant(m.bias.data, 0.0)
        except:
            pass
    return
    
def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    n_triplets = 370;
    n_epochs = end - start
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
                             train=True, transform=None,
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
        #if batch_idx > 0:
        #    continue
        print 'Batch idx', batch_idx
        #print model.detector.shift_net[0].weight.data.cpu().numpy()
        img1, img2, H1to2  = data
        #if np.abs(np.sum(H.numpy()) - 3.0) > 0.01:
        #    continue
        H1to2 = H1to2.squeeze(0)
        do_aug = True
        if torch.abs(H1to2 - torch.eye(3)).sum() > 0.05:
            do_aug = False
        if (img1.size(3) *img1.size(4)   > 1340*1000):
            print img1.shape, ' too big, skipping'
            continue
        img1 = img1.float().squeeze(0)
        img2 = img2.float().squeeze(0)
        if cuda:
            img1, img2, H1to2 = img1.cuda(), img2.cuda(), H1to2.cuda()
        img1, img2, H1to2 = Variable(img1, requires_grad = False), Variable(img2, requires_grad = False), Variable(H1to2, requires_grad = False)
        if do_aug:
            new_img2, H_Orig2New = affineAug(img2, max_add = 0.2 )
            H1to2new = torch.mm(H_Orig2New, H1to2)
        else:
            new_img2 = img2
            H1to2new = H1to2
        #print H1to2
        LAFs1, aff_norm_patches1, resp1, dets1, A1 = HA(img1, True, False, True)
        LAFs2Aug, aff_norm_patches2, resp2, dets2, A2 = HA(new_img2, True, False)
        if (len(LAFs1) == 0) or (len(LAFs2Aug) == 0):
            optimizer.zero_grad()
            continue
        geom_loss, idxs_in1, idxs_in2, LAFs2_in_1  = LAFMagic(LAFs1,
                            LAFs2Aug,
                            H1to2new,
                           3.0, scale_log = 0.3)
        if  len(idxs_in1.size()) == 0:
            optimizer.zero_grad()
            print 'skip'
            continue
        aff_patches_from_LAFs2_in_1 = extract_patches(img1,
                                                      normalizeLAFs(LAFs2_in_1[idxs_in2.long(),:,:], 
                                                      img1.size(3), img1.size(2)))
        SIFTs1 = SIFT(aff_norm_patches1[idxs_in1.long(),:,:,:]).cuda()
        SIFTs2 = SIFT(aff_patches_from_LAFs2_in_1).cuda()
        #sift_snn_loss = loss_HardNet(SIFTs1, SIFTs2, column_row_swap = True,
        #                 margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin");
        patch_dist = 2.0 * torch.sqrt((aff_norm_patches1[idxs_in1.long(),:,:,:]/100. - aff_patches_from_LAFs2_in_1/100.) **2 + 1e-8).view(idxs_in1.size(0),-1).mean(dim = 1)
        sift_dist =  torch.sqrt(((SIFTs1 - SIFTs2)**2 + 1e-12).mean(dim=1))
        loss = geom_loss.cuda() .mean()
        total_loss += loss.data.cpu().numpy()[0]
        #loss += patch_dist
        total_feats += aff_patches_from_LAFs2_in_1.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % 10 == 0:
            print 'A', A1.data.cpu().numpy()[0:1,:,:]
        print 'crafted loss',  pr_l(geom_loss), 'patch', pr_l(patch_dist), 'sift', pr_l(sift_dist)#, 'hardnet',  pr_l(sift_snn_loss)
        print epoch,batch_idx, loss.data.cpu().numpy()[0], idxs_in1.shape

    print 'Train total loss:', total_loss / float(batch_idx+1), ' features ', float(total_feats) / float(batch_idx+1)
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/new_loss_sep_checkpoint_{}.pth'.format(LOG_DIR, epoch))

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
        try:
            LAFs1, aff_norm_patches1, resp1, dets1, A1 = HA(img1, True, False, True)
            LAFs2, aff_norm_patches2, resp2, dets2, A2 = HA(img2, True, False)
        except:
            print 'error, skip'
            continue
        if (len(LAFs1) == 0) or (len(LAFs2) == 0):
            continue
        geom_loss, idxs_in1, idxs_in2, LAFs2_in_1  = LAFMagic(LAFs1,
                            LAFs2,
                            H1to2,
                           5.0, scale_log = 0.4)
        if  len(geom_loss.size()) == 0:
            print 'skip'
            continue
        print 'A', A1.data.cpu().numpy()[0:1,:,:]
        #print fro_dists
        loss = geom_loss.mean()
        total_feats += geom_loss.size(0)
        total_loss += loss.data.cpu().numpy()[0]
        print 'test img', batch_idx, loss.data.cpu().numpy()[0], geom_loss.size(0)
    print 'Total loss:', total_loss / float(batch_idx+1), 'features', float(total_feats) / float(batch_idx+1)
    model.num = model_num_feats

train_loader, test_loader = create_loaders()

HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 350, border = 5, num_Baum_iters = 1, AffNet = LAFNet())
from pytorch_sift import SIFTNet
SIFT = SIFTNet(patch_size = 32 )


if USE_CUDA:
    HA = HA.cuda()

optimizer1 = create_optimizer(HA.AffNet, BASE_LR, 1e-4)

#test(test_loader, model, cuda = USE_CUDA)
for epoch in range(n_epochs):
    print 'epoch', epoch
    train(train_loader, HA, optimizer1, epoch, cuda = USE_CUDA)
    test(test_loader, HA, cuda = USE_CUDA)
