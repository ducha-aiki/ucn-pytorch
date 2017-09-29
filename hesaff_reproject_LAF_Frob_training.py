import torch
import torch.nn as nn
import numpy as np
from  scipy.ndimage import zoom as imzoom
import sys
import os

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


from Utils import GaussianBlur, batch_eig2x2, line_prepender
from LAF import LAFs2ell
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid
from NMS import NMS2d, NMS3dAndComposeA

class HessianAffinePatchExtractor(nn.Module):
    def __init__(self, use_cuda = False, 
                 border = 16,
                 num_features = 500,
                 patch_size = 32,
                 mrSize = 3.0,
                 nlevels = 8,
                 num_Baum_iters = 0,
                 init_sigma = 1.6):
        super(HessianAffinePatchExtractor, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border;
        self.num = num_features
        self.use_cuda = use_cuda
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
        self.Hes = HessianResp()
        self.OriDet = OrientationDetector(patch_size = 19, use_cuda = self.use_cuda);
        self.AffShape = AffineShapeEstimator(patch_size = 19, use_cuda = self.use_cuda)
        self.ScalePyrGen = ScalePyramid(nScales = self.nlevels, init_sigma = self.init_sigma, border = self.b, use_cuda = self.use_cuda)
        return

    def ApplyAffine(self, LAFs, a,b,c):
        A1_ell = torch.cat([a, b], dim = 2)
        A2_ell = torch.cat([b, c], dim = 2)
        A_ell = torch.cat([A1_ell, A2_ell], dim = 1)
        temp_A = torch.bmm(A_ell, LAFs[:,:,0:2])
        return temp_A#torch.cat([temp_A, LAFs[:,:,2:]], dim = 2)
    def rotateLAFs(self, LAFs, angles):
        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        A1_ang = torch.cat([cos_a, sin_a], dim = 2)
        A2_ang = torch.cat([-sin_a, cos_a], dim = 2)
        A_ang = torch.cat([A1_ang, A2_ang], dim = 1)
        temp_A = torch.bmm(LAFs[:,:,0:2], A_ang )
        return torch.cat([temp_A, LAFs[:,:,2:]], dim = 2)
    def extract_patches(self, scale_pyramid, LAFs, pyr_idxs, level_idxs, PS = 19, gauss_mask = False, use_cuda = False, pyr_inv_idxs = None):
        patches_list = []
        if gauss_mask:
            mask = torch.from_numpy(CircularGaussKernel(kernlen = PS, circ_zeros = False).astype(np.float32))
            mask = Variable(mask)
            if use_cuda:
                mask = mask.cuda()
        if pyr_inv_idxs is not None:
            for i in range(len(scale_pyramid)):
                for j in range(1, len(level_idxs) - 1):
                    cur_lvl_idxs = pyr_inv_idxs[i][j]
                    if cur_lvl_idxs is None:
                        continue
                    curr_aff = LAFs[cur_lvl_idxs.view(-1), :,:]
                    grid = torch.nn.functional.affine_grid(curr_aff, torch.Size((cur_lvl_idxs.size(0),
                                                                    1,
                                                                    PS, 
                                                                    PS)))
                    patches_list.append(torch.nn.functional.grid_sample(scale_pyramid[i][j].expand(curr_aff.size(0),
                                                                                scale_pyramid[i][0].size(1), 
                                                                                scale_pyramid[i][0].size(2), 
                                                                                scale_pyramid[i][0].size(3)),  grid))
        else: #Calculate inverted indexes, but it is slow
            for i in range(len(scale_pyramid)):
                cur_idxs = pyr_idxs == i #torch.nonzero((pyr_idxs == i).data)
                for j in range(1, len(level_idxs) - 1):
                    cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
                    #cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
                    if len(cur_lvl_idxs.size()) == 0:
                        continue
                    ##LAFs[cur_lvl_idxs, :,:]
    #               print curr_aff.shape
                    curr_aff = LAFs[cur_lvl_idxs.view(-1), :,:]
                    grid = torch.nn.functional.affine_grid(curr_aff, torch.Size((cur_lvl_idxs.size(0),
                                                                    1,
                                                                    PS, 
                                                                    PS)))
                    patches_list.append(torch.nn.functional.grid_sample(scale_pyramid[i][j].expand(curr_aff.size(0),
                                                                                scale_pyramid[i][0].size(1), 
                                                                                scale_pyramid[i][0].size(2), 
                                                                                scale_pyramid[i][0].size(3)),  grid))
        
        patches = torch.cat(patches_list, dim = 0)
        if gauss_mask:
            patches = patches * mask.unsqueeze(0).unsqueeze(0).expand(patches.size(0),1,PS,PS)
        return patches
    def forward(self,x):
        ### Generate scale space
        scale_pyr, sigmas, pix_dists = self.ScalePyrGen(x)
        ### Detect keypoints in scale space
        aff_matrices = []
        top_responces = []
        pyr_idxs = []
        level_idxs = []
        for oct_idx in range(len(sigmas)):
            #print oct_idx
            octave = scale_pyr[oct_idx]
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            for level_idx in range(1,len(octave)-1):
                low = float(sigmas_oct[level_idx - 1 ]**4) * self.Hes(octave[level_idx - 1])
                cur = float(sigmas_oct[level_idx]**4) * self.Hes(octave[level_idx])
                high = float(sigmas_oct[level_idx + 1 ]**4) * self.Hes(octave[level_idx + 1])
                nms_f = NMS3dAndComposeA(scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                         mrSize = 1.0,
                                        border = self.b, use_cuda = self.use_cuda)
                top_resp, aff_matrix = nms_f(low,cur,high, self.num / 2)
                
                aff_matrices.append(aff_matrix), top_responces.append(top_resp)
                pyr_id = Variable(oct_idx * torch.ones(aff_matrix.size(0)))
                lev_id = Variable(level_idx * torch.ones(aff_matrix.size(0)))
                if self.use_cuda:
                    pyr_id = pyr_id.cuda()
                    lev_id = lev_id.cuda()
                pyr_idxs.append(pyr_id)
                level_idxs.append(lev_id)
        top_resp_scales = torch.cat(top_responces, dim = 0)
        aff_m_scales = torch.cat(aff_matrices,dim = 0)
        pyr_idxs_scales = torch.cat(pyr_idxs,dim = 0)
        level_idxs_scale = torch.cat(level_idxs, dim = 0)
        #print top_resp_scales
        final_resp, idxs = torch.topk(top_resp_scales, k = max(1, min(self.num, top_resp_scales.size(0))));
        final_aff_m = torch.index_select(aff_m_scales, 0, idxs)
        final_pyr_idxs = torch.index_select(pyr_idxs_scales,0,idxs)
        final_level_idxs = torch.index_select(level_idxs_scale,0,idxs)
        ###
        pyr_inv_idxs = []
        ### Precompute octave inverted indexes
        for i in range(len(scale_pyr)):
            pyr_inv_idxs.append([])
            cur_idxs = final_pyr_idxs == i #torch.nonzero((pyr_idxs == i).data)
            for j in range(0, len(final_level_idxs)):
                cur_lvl_idxs = torch.nonzero(((final_level_idxs == j) * cur_idxs).data)
                if len(cur_lvl_idxs.size()) == 0:
                    pyr_inv_idxs[-1].append(None)
                else:
                    pyr_inv_idxs[-1].append(cur_lvl_idxs.squeeze(1))
        ###
        #final_aff_m[:,:,0:2] =  final_aff_m[:,:,0:2] / self.init_sigma
        patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs,final_level_idxs, PS = 19, gauss_mask = False, pyr_inv_idxs = pyr_inv_idxs);
        ###
        base_A = Variable(torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2))
        if self.use_cuda:
            base_A = base_A.cuda()
        ### Estimate affine shape
        for i in range(self.num_Baum_iters):
            #print i
            a,b,c,ratio_in_patch = self.AffShape(patches_small)
            base_A_new = self.ApplyAffine(base_A, a,b,c)
            l1,l2 = batch_eig2x2(base_A_new)
            ratio = torch.abs(l1 / (l2 + 1e-8))
            mask = (ratio <= 6.0) * (ratio >= 1./6.)
            #print mask.sum()
            mask = mask.unsqueeze(1).unsqueeze(1).float().expand(mask.size(0),2,2)
            base_A = base_A_new * mask + base_A * (1.0 - mask)
            #idxs_mask = mask.data.nonzero().view(-1)
            #base_A = base_A_new[idxs_mask,:,:]
            #final_aff_m = final_aff_m[idxs_mask, :, :]
            #final_pyr_idxs = final_pyr_idxs[idxs_mask]
            
            temp_final = torch.cat([torch.bmm(base_A,final_aff_m[:,:,:2]), final_aff_m[:,:,2:] ], dim =2)
            if i != self.num_Baum_iters - 1:
                patches_small = self.extract_patches(scale_pyr, temp_final, final_pyr_idxs, final_level_idxs, PS = 19, gauss_mask = False, pyr_inv_idxs = pyr_inv_idxs);
            #else:
            #    idxs_mask = torch.nonzero(((ratio <= 6.0) * (ratio >= 1./6.)).data).view(-1)
            #    temp_final = temp_final[idxs_mask, :, :]
            #    final_pyr_idxs = final_pyr_idxs[idxs_mask]
            #    final_level_idxs = final_level_idxs[idxs_mask]
        #
        if self.num_Baum_iters > 0:
            final_aff_m = temp_final
        #####
        #final_aff_m[:,:,0:2] = self.init_sigma * self.mrSize * final_aff_m[:,:,0:2]
        final_aff_m[:,:,0:2] =  self.mrSize * final_aff_m[:,:,0:2]
        patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs, PS = 19, gauss_mask = False, pyr_inv_idxs = pyr_inv_idxs);
        ######
        ### Detect orientation
        for i in range(0):
            ori = self.OriDet(patches_small)
            #print np.degrees(ori.data.cpu().numpy().ravel()[1])
            #print final_aff_m[1,:,:]
            #print '*****'
            final_aff_m = self.rotateLAFs(final_aff_m, ori)
            #print final_aff_m[0,:,:]
            patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs,  PS = 19, gauss_mask = False, pyr_inv_idxs = pyr_inv_idxs);
        ###
        patches = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs, PS = self.PS, pyr_inv_idxs = pyr_inv_idxs);
        #scale back to image scale
        final_aff_m[:,0,2] = final_aff_m[:,0,2] * x.size(3)
        final_aff_m[:,1,2] = final_aff_m[:,1,2] * x.size(2)
        min_shape = min(float(x.size(2)),float(x.size(3)))
        final_aff_m[:,:,0:2]  = final_aff_m[:,:,0:2] * min_shape
        return final_aff_m,patches,final_resp,scale_pyr
    

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
    print min_dist.min(), min_dist.max(), min_dist.mean()
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
        shuffle = False, **kwargs)

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
        if (img1.size(3) *img1.size(4)   > 1200*800):
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
        fro_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes_Fro(LAFs1, LAFs2, H, dist_threshold = 0.1, use_cuda = cuda);
        if  len(fro_dists.size()) == 0:
            optimizer.zero_grad()
            print 'skip'
            continue
        loss = fro_dists.mean()
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
        fro_dists, idxs_in1, idxs_in2 = get_GT_correspondence_indexes_Fro(LAFs1, LAFs2, H, dist_threshold = 100, use_cuda = cuda);
        loss = fro_dists.mean()
        total_loss += loss.data.cpu().numpy()[0]
        print 'test img', batch_idx, loss.data.cpu().numpy()[0]
    print 'Total loss:', total_loss / float(batch_idx+1)

train_loader, test_loader = create_loaders()

HA = HessianAffinePatchExtractor( mrSize = 5.196, num_features = 4000, border = 5, num_Baum_iters = 5, use_cuda = USE_CUDA)


model = HA
if USE_CUDA:
    model = model.cuda()

optimizer1 = create_optimizer(model, BASE_LR, 5e-5)


start = 0
end = 100
for epoch in range(start, end):
    print 'epoch', epoch
    if USE_CUDA:
        model = model.cuda()
    test(test_loader, model, cuda = USE_CUDA)
    train(train_loader, model, optimizer1, epoch, cuda = USE_CUDA)
test(test_loader, model, cuda = USE_CUDA)
