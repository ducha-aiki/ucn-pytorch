import torch
import torch.nn as nn
import numpy as np
from  scipy.ndimage import zoom as imzoom
import sys
import os
%matplotlib inline

from PIL import Image
from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np

#Training....
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm





from SpatialTransformer2D import SpatialTransformer2d
from HardNet import HardNet
hardnet = HardNet()
checkpoint = torch.load('HardNetLib.pth')
hardnet.load_state_dict(checkpoint['state_dict'])

class SparseImgRepresenter(nn.Module):
    def __init__(self, 
             detector_net = None,
             descriptor_net = None,    
             use_cuda = False):
        super(SparseImgRepresenter, self).__init__()
        self.detector = detector_net;
        self.descriptor = descriptor_net;
        return
    def forward(self, input_img):
        aff_norm_patches, LAFs = self.detector(input_img)
        descs = self.descriptor(aff_norm_patches);
        return aff_norm_patches, LAFs, descs

ConvST_net = SpatialTransformer2d( num_input_channels = 1,
                 feature_net = None,
                 out_patch_size = 32,
                 out_stride = 32,
                 min_zoom = 0.7,
                 max_zoom = 1.3,
                 min_tilt = 0.8,
                 max_tilt = 1.2,
                 max_rot = 1.0,
                 max_shift = 0.1,
                 mrSize = 1.0, use_cuda = True)

SIRNet = SparseImgRepresenter(detector_net = ConvST_net, descriptor_net = hardnet)

#aff_norm_patches, LAFs, descs = SIRNet(var_image_reshape)

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""

    d1_sq = torch.sum(anchor * anchor, dim=1)
    d2_sq = torch.sum(positive * positive, dim=1)
    eps = 1e-6
    return torch.sqrt((d1_sq.expand(anchor.size(0), anchor.size(0)) + torch.t(d2_sq.expand(positive.size(0), positive.size(0)))
                      - 2.0 * torch.bmm(positive.unsqueeze(0), torch.t(anchor).unsqueeze(0)).squeeze(0))+eps)
def LAFs_to_H_frames(aff_pts, use_cuda = False):
    H3_x = torch.Tensor([0, 0, 1 ]).unsqueeze(0).unsqueeze(0).expand_as(aff_pts[:,0:1,:]);
    H3_x = torch.autograd.Variable(H3_x)
    if use_cuda:
        H3_x = H3_x.cuda()
    return torch.cat([aff_pts, H3_x], dim = 1)
def get_GT_correspondence_indexes(aff_pts1,aff_pts2, H1to2, dist_threshold = 4, use_cuda = False):
    LHF2 = LAFs_to_H_frames(aff_pts2, use_cuda = use_cuda)
    LHF2_reprojected_to_1 = torch.bmm(H1to2.unsqueeze(0).expand_as(LHF2), LHF2);
    LHF2_reprojected_to_1 = LHF2_reprojected_to_1 / LHF2_reprojected_to_1[:,2:,2:].expand_as(LHF2_reprojected_to_1);
    just_centers1 = aff_pts1[:,:,2];
    just_centers2_repr_to_1 = LHF2_reprojected_to_1[:,0:2,2];
    dist  = distance_matrix_vector(just_centers1, just_centers2_repr_to_1)
    min_dist, idxs_in_2 = torch.min(dist,1)
    plain_indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)))
    if use_cuda:
        plain_indxs_in1 = plain_indxs_in1.cuda()
    mask =  min_dist <= dist_threshold
    return min_dist[mask], plain_indxs_in1[mask], idxs_in_2[mask]



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
        group['lr'] =  0.01 * (
        1.0 - float(group['step']) * float(1.0) / (n_triplets * float(n_epochs)))
    return

def create_optimizer(model, new_lr, wd):
    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=new_lr,
                          momentum=0.9, dampening=0.9,
                          weight_decay=wd)
    return optimizer

def create_loaders(load_random_triplets = False):

    kwargs = {'num_workers': 2, 'pin_memory': True} if True else {}

    transform = transforms.Compose([
            transforms.ToTensor()])
    #        transforms.Normalize((args.mean_image,), (args.std_image,))])

    train_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/dev/LearnedDetector/dataset', 'a',
                             train=True, transform=None, 
                             download=True), batch_size = 1,
        shuffle = True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            dset.HPatchesSeq('/home/old-ufo/dev/LearnedDetector/dataset', 'a',
                             train=False, transform=None, 
                             download=True), batch_size = 1,
        shuffle = False, **kwargs)

    return train_loader, test_loader


def train(train_loader, model, optimizer, epoch, cuda = True):
    # switch to train mode
    model.train()
    log_interval = 1
    spatial_only = True
    pbar = tqdm(enumerate(train_loader))
    loss_function =  nn.MSELoss()
    for batch_idx, data in pbar:
        img1, img2, H  = data
        img1 = img1.float().squeeze(0)
        img2 = img2.float().squeeze(0)
        if cuda:
            img1, img2, H = img1.cuda(), img2.cuda(), H.cuda()
        img1, img2, H = Variable(img1), Variable(img2), Variable(H)
        aff_norm_patches1, LAFs1, descs1 = model(img1)
        aff_norm_patches2, LAFs2, descs2 = model(img2)
        spatial_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes(LAFs1, LAFs2, H, dist_threshold = 10, use_cuda = cuda);
        if spatial_only:
            target =  Variable(torch.zeros_like(spatial_dists).cuda())
            loss = loss_function(spatial_dists,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        if batch_idx % alog_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0]))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))


model = SIRNet
train_loader, test_loader = create_loaders()
model = model.cuda()

optimizer1 = create_optimizer(model.detector, 0.01, 1e-4)

start = 0
end = 10
for epoch in range(start, end):
    print 'epoch', epoch
    model = model.cuda()
    train(train_loader, model, optimizer1, epoch, cuda = True)