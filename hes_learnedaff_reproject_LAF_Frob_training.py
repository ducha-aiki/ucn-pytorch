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
BASE_LR = 0.01
from SpatialTransformer2D import SpatialTransformer2d
from HardNet import HardNet
#hardnet = HardNet()
#checkpoint = torch.load('HardNetLib.pth')
#hardnet.load_state_dict(checkpoint['state_dict'])


from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A


class BaumNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(BaumNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 3, kernel_size=4, bias = True),
            nn.Tanh()
        )
        self.features.apply(weights_init)
        return
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        abc = self.features(self.input_norm(input))
        return abc2A(abc[:,0,:,:] + 1. ,abc[:,1,:,:] , abc[:,2,:,:] + 1.):

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=1.0)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
    

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)
    eps = 1e-6
    return torch.sqrt(torch.abs((d1_sq.expand(positive.size(0), anchor.size(0)) +
                       torch.t(d2_sq.expand(anchor.size(0), positive.size(0)))
                      - 2.0 * torch.bmm(positive.unsqueeze(0), torch.t(anchor).unsqueeze(0)).squeeze(0))+eps))
def LAFs_to_H_frames(aff_pts, use_cuda = False):
    H3_x = torch.Tensor([0, 0, 1 ]).unsqueeze(0).unsqueeze(0).expand_as(aff_pts[:,0:1,:]);
    H3_x = torch.autograd.Variable(H3_x)
    if use_cuda:
        H3_x = H3_x.cuda()
    return torch.cat([aff_pts, H3_x], dim = 1)
def reproject_to_canonical_Frob_batched(LHF1_inv, LHF2, batch_size = 2, use_cuda = False):
    out = torch.autograd.Variable(torch.zeros((LHF1_inv.size(0), LHF2.size(0))))
    eye1 = torch.autograd.Variable(torch.eye(3), requires_grad = False)
    if use_cuda:
        out = out.cuda()
        eye1 = eye1.cuda()
    len1 = LHF1_inv.size(0)
    len2 = LHF2.size(0)
    #print LHF1_inv
    #print LHF2 
    n_batches = int(np.floor(len1 / batch_size) + 1);
    for b_idx in range(n_batches):
        #print b_idx
        start = b_idx * batch_size;
        fin = min((b_idx+1) * batch_size, len1)
        current_bs = fin - start
        if current_bs == 0:
            break
        should_be_eyes = torch.bmm(LHF1_inv[start:fin, :, :].unsqueeze(0).expand(len2,current_bs, 3, 3).contiguous().view(-1,3,3),
                                   LHF2.unsqueeze(1).expand(len2,current_bs, 3,3).contiguous().view(-1,3,3))
        out[start:fin, :] = torch.sum((should_be_eyes - eye1.unsqueeze(0).expand_as(should_be_eyes))**2, dim=1).sum(dim = 1).view(current_bs, len2)
    return out

def get_GT_correspondence_indexes(aff_pts1,aff_pts2, H1to2, dist_threshold = 4, use_cuda = False):
    LHF2 = LAFs_to_H_frames(aff_pts2, use_cuda = use_cuda)
    LHF2_reprojected_to_1 = torch.bmm(H1to2.expand_as(LHF2), LHF2);
    LHF2_reprojected_to_1 = LHF2_reprojected_to_1 / LHF2_reprojected_to_1[:,2:,2:].expand_as(LHF2_reprojected_to_1);
    just_centers1 = aff_pts1[:,:,2];
    just_centers2_repr_to_1 = LHF2_reprojected_to_1[:,0:2,2];
    dist  = distance_matrix_vector(just_centers2_repr_to_1, just_centers1)
    min_dist, idxs_in_2 = torch.min(dist,1)
    plain_indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)),requires_grad = False)
    if use_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]

def get_GT_correspondence_indexes_Fro(aff_pts1,aff_pts2, H1to2, dist_threshold = 4, use_cuda = False):
    LHF2 = LAFs_to_H_frames(aff_pts2, use_cuda = use_cuda)
    LHF2_reprojected_to_1 = torch.bmm(H1to2.expand_as(LHF2), LHF2);
    LHF2_reprojected_to_1 = LHF2_reprojected_to_1 / LHF2_reprojected_to_1[:,2:,2:].expand_as(LHF2_reprojected_to_1);
    LHF1 = LAFs_to_H_frames(aff_pts1, use_cuda = use_cuda)
    
    LHF1_inv = torch.autograd.Variable(torch.zeros(LHF1.size()))
    if use_cuda:
        LHF1_inv = LHF1_inv.cuda()
    for i in range(len(LHF1_inv)):
        LHF1_inv[i,:,:] = LHF1[i,:,:].inverse()
    frob_norm_dist = reproject_to_canonical_Frob_batched(LHF1_inv, LHF2_reprojected_to_1, batch_size = 2, use_cuda = use_cuda)
    min_dist, idxs_in_2 = torch.min(frob_norm_dist,1)
    plain_indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)
    if use_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    #print min_dist.min(), min_dist.max(), min_dist.mean()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]
#
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
        group['lr'] =  BASE_LR #*  .0 - float(group['step']) * float(1.0) / (n_triplets * float(n_epochs)))
    return

def create_optimizer(model, new_lr, wd):
    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                          momentum=0.5, dampening=0.5,
                          weight_decay=wd)
    return optimizer

def create_loaders(load_random_triplets = False):

    kwargs = {'num_workers': 2, 'pin_memory': True} if True else {}

    transform = transforms.Compose([
            transforms.ToTensor()])
    #        transforms.Normalize((args.mean_image,), (args.std_image,))])

    train_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/storage/learned_detector/dataset/', 'a',
                             train=True, transform=None, 
                             download=True), batch_size = 1,
        shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/storage/learned_detector/dataset/', 'a',
                             train=False, transform=None, 
                             download=True), batch_size = 1,
        shuffle = False, **kwargs)

    return train_loader, test_loader

def train(train_loader, model, optimizer, epoch, cuda = True):
    # switch to train mode
    model.train()
    log_interval = 1
    spatial_only = True
    pbar = enumerate(train_loader)
    for batch_idx, data in pbar:
        print 'Batch idx', batch_idx
        #print model.detector.shift_net[0].weight.data.cpu().numpy()
        img1, img2, H  = data
        #if np.abs(np.sum(H.numpy()) - 3.0) > 0.01:
        #    continue
        H = H.squeeze(0)
        if (img1.size(3) *img1.size(4)   > 1340*1000):
            print img1.shape, ' too big, skipping'
            continue
        img1 = img1.float().squeeze(0)
        #img1 = img1 - img1.mean()
        #img1 = img1 / 50.#(img1.std() + 1e-8)
        img2 = img2.float().squeeze(0)
        #img2 = img2 - img2.mean()
        #img2 = img2 / 50.#(img2.std() + 1e-8)
        if cuda:
            img1, img2, H = img1.cuda(), img2.cuda(), H.cuda()
        img1, img2, H = Variable(img1, requires_grad = False), Variable(img2, requires_grad = False), Variable(H, requires_grad = False)
        LAFs1, aff_norm_patches1, resp1, pyr1 = HA(img1 / 255.)
        LAFs2, aff_norm_patches2, resp2, pyr2 = HA(img2 / 255.)
        if (len(LAFs1) == 0) or (len(LAFs2) == 0):
            optimizer.zero_grad()
            LAFs1,aff_norm_patches1,resp1,pyr1 = None,None,None,None
            LAFs2,aff_norm_patches2,resp2,pyr2 = None,None,None,None
            continue
        fro_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes_Fro(LAFs1, LAFs2, H, dist_threshold = 10., use_cuda = cuda);
        if  len(fro_dists.size()) == 0:
            LAFs1,aff_norm_patches1,resp1,pyr1 = None,None,None,None
            LAFs2,aff_norm_patches2,resp2,pyr2 = None,None,None,None
            fro_dists, idxs_in1, idxs_in2 = None,None,None
            optimizer.zero_grad()
            print 'skip'
            continue
        loss = fro_dists.mean()
        patch_dist = torch.mean((aff_norm_patches1[idxs_in1.data.long(),:,:,:] - aff_norm_patches2[idxs_in2.data.long(), :,:,:]) **2)
        print loss.data.cpu().numpy()[0], patch_dist.data.cpu().numpy()[0]
        loss += patch_dist
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #adjust_learning_rate(optimizer)
        print epoch,batch_idx, loss.data.cpu().numpy()[0], idxs_in1.shape

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def test(test_loader, model, cuda = True):
    # switch to train mode
    model.eval()
    log_interval = 1
    pbar = enumerate(train_loader)
    total_loss = 0
    for batch_idx, data in pbar:
        print 'Batch idx', batch_idx
        img1, img2, H  = data
        if (img1.size(3) *img1.size(4)   > 1500*1200):
            print img1.shape, ' too big, skipping'
            continue
        H = H.squeeze(0)
        img1 = img1.float().squeeze(0)
        #img1 = img1 - img1.mean()
        #img1 = img1 / 50.#(img1.std() + 1e-8)
        img2 = img2.float().squeeze(0)
        #img2 = img2 - img2.mean()
        #img2 = img2 / 50.#(img2.std() + 1e-8)
        if cuda:
            img1, img2, H = img1.cuda(), img2.cuda(), H.cuda()
        img1, img2, H = Variable(img1, volatile = True), Variable(img2, volatile = True), Variable(H, volatile = True)
        LAFs1, aff_norm_patches1, resp1, pyr1 = HA(img1 / 255.)
        LAFs2, aff_norm_patches2, resp2, pyr2 = HA(img2 / 255.)
        if (len(LAFs1) == 0) or (len(LAFs2) == 0):
            continue
        fro_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes_Fro(LAFs1, LAFs2, H, dist_threshold = 10, use_cuda = cuda);
        if  len(fro_dists.size()) == 0:
            print 'skip'
            continue
        loss = fro_dists.mean()
        total_loss += loss.data.cpu().numpy()[0]
        print 'test img', batch_idx, loss.data.cpu().numpy()[0]
    print 'Total loss:', total_loss / float(batch_idx+1)

train_loader, test_loader = create_loaders()

HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.0, num_features = 3000, border = 1, num_Baum_iters = 5, AffNet = BaumNet())


model = HA
if USE_CUDA:
    model = model.cuda()

optimizer1 = create_optimizer(model.AffShape, BASE_LR, 5e-5)


start = 0
end = 100
for epoch in range(start, end):
    print 'epoch', epoch
    if USE_CUDA:
        model = model.cuda()
    train(train_loader, model, optimizer1, epoch, cuda = USE_CUDA)
    test(test_loader, model, cuda = USE_CUDA)
test(test_loader, model, cuda = USE_CUDA)
