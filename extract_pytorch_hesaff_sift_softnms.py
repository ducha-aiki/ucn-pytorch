import torch
import torch.nn as nn
import numpy as np
from  scipy.ndimage import zoom as imzoom
import sys
import os

import seaborn as sns
import time
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
import math
import torch.nn.functional as F
USE_CUDA = False
from copy import deepcopy

LOG_DIR = 'log_snaps'
BASE_LR = 0.00000001
from SpatialTransformer2D import SpatialTransformer2d
from HardNet import HardNet
from Utils import CircularGaussKernel

hardnet = HardNet()
checkpoint = torch.load('HardNetLib.pth')
hardnet.load_state_dict(checkpoint['state_dict'])

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print "Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py imgs/ref.png out.txt"
    sys.exit(1)


img = Image.open(input_img_fname).convert('RGB')
img = np.mean(np.array(img), axis = 2)

var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)))
var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
def generate_2dgrid(h,w, centered = True):
    if centered:
        x = torch.linspace(-w/2+1, w/2, w)
        y = torch.linspace(-h/2+1, h/2, h)
    else:
        x = torch.linspace(0, w-1, w)
        y = torch.linspace(0, h-1, h)
    grid2d = torch.stack([y.repeat(w,1).t().contiguous().view(-1), x.repeat(h)],1)
    return grid2d
def generate_3dgrid(d, h, w, centered = True):
    if type(d) is not list:
        if centered:
            z = torch.linspace(-d/2+1, d/2, d)
        else:
            z = torch.linspace(0, d-1, d)
        dl = d
    else:
        z = torch.FloatTensor(d)
        dl = len(d)
    grid2d = generate_2dgrid(h,w, centered = centered)
    grid3d = torch.cat([z.repeat(w*h,1).t().contiguous().view(-1,1), grid2d.repeat(dl,1)],dim = 1)
    return grid3d

def zero_response_at_border(x, b):
    x[:, :,  0:b, :] =  0
    x[:, :,  x.size(2)-b: , :] =  0

    x[:, :, :,  0:b] =  0
    x[:, :, :,   x.size(3) - b:] =  0
    return x
class NMS3dAndComposeA(nn.Module):
    def __init__(self, mrSize = 1.0, kernel_size = 3, threshold = 0, use_cuda = False, scales = None, border = 3):
        super(NMS3dAndComposeA, self).__init__()
        self.mrSize = mrSize;
        self.eps = 1e-5
        self.ks = 3
        if type(scales) is not list:
            self.grid = generate_3dgrid(3,self.ks,self.ks)
        else:
            self.grid = generate_3dgrid(scales,self.ks,self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3,3,3,3))
        self.th = threshold
        self.use_cuda = use_cuda
        self.cube_idxs = []
        self.border = border
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3,3,3,3))
        if self.use_cuda:
            self.grid = self.grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
        return
    def forward(self, low, cur, high, num_feats = 500):
        spatial_grid = Variable(generate_2dgrid(low.size(2), low.size(3), False)).view(1,low.size(2), low.size(3),2)
        spatial_grid = spatial_grid.permute(3,1, 2, 0)
        if self.use_cuda:
            spatial_grid = spatial_grid.cuda()
        resp3d = torch.cat([low,cur,high], dim = 1)
        exp_resp3d = torch.exp(torch.sqrt(resp3d + 1e-8) * self.beta)
        
        #residual_to_patch_center
        softargmax3d = F.conv2d(exp_resp3d,
                                self.grid,
                                padding = 1) / (F.conv2d(exp_resp3d, self.grid_ones, padding = 1) + 1e-8)
        
        #maxima coords
        softargmax3d[:,1:,:,:] = softargmax3d[:,1:,:,:] + spatial_grid
        sc_y_x = softargmax3d.view(3,-1).t()
        
        mask = (cur > low)*(cur > high) *((cur - F.max_pool2d(cur,kernel_size = 3,
                                                            padding = 1,
                                                            stride = 1) + self.eps) > 0)
        mask = zero_response_at_border(mask, self.border)
        nmsed_resp_flat = (mask.float() * cur).view(-1)
        topk_val, idxs = torch.topk(nmsed_resp_flat, 
                                    k = max(1, min(int(num_feats), nmsed_resp_flat.size(0))));
        
        sc_y_x_topk = sc_y_x[idxs.data,:]
        
        sc_y_x_topk[:,1] = sc_y_x_topk[:,1] / float(cur.size(2))
        sc_y_x_topk[:,2] = sc_y_x_topk[:,2] / float(cur.size(3))
        
        min_size = float(min((cur.size(2)), cur.size(3)))
        base_A = Variable(self.mrSize * torch.eye(2).unsqueeze(0).expand(idxs.size(0),2,2).float() / min_size)
        if self.use_cuda:
            base_A = base_A.cuda()
        A = sc_y_x_topk[:,:1].unsqueeze(1).expand_as(base_A) * base_A
        full_A  = torch.cat([A,
                             torch.cat([sc_y_x_topk[:,2:].unsqueeze(-1),
                                        sc_y_x_topk[:,1:2].unsqueeze(-1)], dim=1)], dim = 2)
        return topk_val, full_A
class SparseImgRepresenter(nn.Module):
    def __init__(self, 
             detector_net = None,
             descriptor_net = None,    
             use_cuda = False):
        super(SparseImgRepresenter, self).__init__()
        self.detector = detector_net;
        self.descriptor = descriptor_net;
        return
    def forward(self, input_img, skip_desc = False):
        aff_norm_patches, LAFs = self.detector(input_img)
        if not skip_desc:
            descs = self.descriptor(aff_norm_patches);
            return aff_norm_patches, LAFs, descs
        return aff_norm_patches, LAFs
def LAF2pts(LAF, n_pts = 50):
    a = np.linspace(0, 2*np.pi, n_pts);
    x = list(np.cos(a))
    x.append(0)
    x = np.array(x).reshape(1,-1)
    y = list(np.sin(a))
    y.append(0)
    y = np.array(y).reshape(1,-1)
    
    HLAF = np.concatenate([LAF, np.array([0,0,1]).reshape(1,3)])
    H_pts =np.concatenate([x,y,np.ones(x.shape)])
    #print H_pts.shape, HLAF.shape
    H_pts_out = np.transpose(np.matmul(HLAF, H_pts))#np.tensordot(HLAF,H_pts, axes = 0)
    H_pts_out[:,0] = H_pts_out[:,0] / H_pts_out[:, 2]
    H_pts_out[:,1] = H_pts_out[:,1] / H_pts_out[:, 2]
    return H_pts_out[:,0:2]
def visualize_LAFs(img, LAFs):
    plt.figure()
    plt.imshow(255 - img)
    min_shape = min(float(img.shape[1]),float(img.shape[0]))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        LAF[:,:2] *= min_shape
        LAF[0,2] *= float(img.shape[1])
        LAF[1,2] *= float(img.shape[0])
        #print LAF
        ell = LAF2pts(LAF)
        plt.plot( ell[:,0], ell[:,1], 'r')
    plt.show()
    return
def convert_LAF_to_this_stupid_Oxford_ellipse_format(img, LAFs):
    print img.shape
    h,w = img.shape
    min_shape = min(h,w)
    ellipses = np.zeros((len(LAFs),5))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        LAF[0,2] *= float(img.shape[1])
        LAF[1,2] *= float(img.shape[0])
        scale = np.sqrt(LAF[0,0]*LAF[1,1]  - LAF[0,1]*LAF[1, 0] + 1e-10)
        LAF[0:2,0:2] /=  scale;
        scale *= float(min_shape)
        u, W, v = np.linalg.svd(LAF[0:2,0:2], full_matrices=True)
        W[0] = 1. / (W[0]*W[0]*scale*scale)
        W[1] = 1. / (W[1]*W[1]*scale*scale)
        A =  np.matmul(np.matmul(u, np.diag(W)), u.transpose())
        ellipses[i,0] = LAF[0,2]
        ellipses[i,1] = LAF[1,2]
        ellipses[i,2] = A[0,0]
        ellipses[i,3] = A[0,1]
        ellipses[i,4] = A[1,1]
    return ellipses

class HessianResp(nn.Module):
    def __init__(self):
        super(HessianResp, self).__init__()
        
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), padding = (0,1), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), padding = (1,0), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.gxx =  nn.Conv2d(1, 1, kernel_size=(1,3), padding = (0,1), bias = False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0]]]], dtype=np.float32))
        
        self.gyy =  nn.Conv2d(1, 1, kernel_size=(3,1), padding = (1,0), bias = False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0], [1.0]]]], dtype=np.float32))
        
        return
    def forward(self, x):
        gxx = self.gxx(x)
        gyy = self.gyy(x)
        gxy = self.gy(self.gx(x))
        return torch.abs(gxx * gyy - gxy * gxy)
        
class NMS2d(nn.Module):
    #Outputs coordinatas 
    def __init__(self, kernel_size = 3, threshold = 0, use_cuda = False):
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False, padding = 1)
        self.eps = 1e-5
        self.th = threshold
        self.use_cuda = use_cuda
        return
    def forward(self, x):
        ttt = self.MP(x)
        if self.th > self.eps:
            return  x *(ttt > self.th).float() * ((torch.abs(x) + self.eps - ttt) > 0).float()
        else:
            return ((x - ttt + self.eps) > 0).float() * x



class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.6):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        return
    def calculate_weights(self,  sigma):
        kernel = CircularGaussKernel(sigma = sigma, circ_zeros = False)
        h,w = kernel.shape
        halfSize = float(h) / 2.;
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w);
    def forward(self, x):
        return F.conv2d(x, Variable(self.buf), padding = self.pad)


def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = int(round(2.0 * math.floor(patch_size / 2) / float(num_spatial_bins + 1)))
    bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
    return bin_weight_kernel_size, bin_weight_stride


class AffineShapeEstimator(nn.Module):
    def __init__(self, use_cuda = False, 
                 threshold = 0.001, patch_size = 19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold;
        self.use_cuda = use_cuda;
        self.PS = patch_size
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), padding = (0,1), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))
        
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), padding = (1,0), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))
        
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=patch_size, circ_zeros = True).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        if use_cuda:
            self.gk = self.gk.cuda()
        return
    def invSqrt(self,a,b,c):
        eps = 1e-10
        r1 = (b != 0).float() * (c - a) / (2. * b + eps) + 0.
        t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1. + r1*r1));
        r = 1.0 / torch.sqrt( 1. + t1*t1)
        t = t1*r;
        
        r = r * (b != 0).float() + 1.0 * (b == 0).float();
        t = t * (b != 0).float() + 0. * (b == 0).float();
        
        x = 1. / torch.sqrt( r*r*a - 2*r*t*b + t*t*c + eps)
        z = 1. / torch.sqrt( t*t*a + 2*r*t*b + r*r*c + eps)
        
        d = torch.sqrt( x * z)
        
        x = x / d
        z = z / d
        
        l1 = torch.max(x,z)
        l2 = torch.min(x,z)
        
        new_a = r*r*x + t*t*z
        new_b = -r*t*x + t*r*z
        new_c = t*t*x + r*r *z

        return l1,l2, new_a, new_b, new_c
    def forward(self,x):
        gx = self.gx(x)
        gy = self.gy(x)
        a1 = (gx*gx * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        b1 = (gx*gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        c1 = (gy*gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        l1,l2,a, b, c = self.invSqrt(a1,b1,c1)
        rat1 = l1/l2
        eig_ratio = 1. - 1./rat1;
        
        den = torch.sqrt(a*c - b*b)
        mask = (rat1 <= 2).float().view(-1);
        a = a * mask / den + 1. * (1.- mask)
        b = b * mask / den + 0. * (1.- mask)
        c = c * mask / den + 1. * (1.- mask)
        return a.view(-1,1,1),b.view(-1,1,1),c.view(-1,1,1), eig_ratio
    
class OrientationDetector(nn.Module):
    def __init__(self, use_cuda = False,
                mrSize = 3.0, patch_size = None):
        super(OrientationDetector, self).__init__()
        if patch_size is None:
            patch_size = 32;
        self.bin_weight_kernel_size, self.bin_weight_stride = get_bin_weight_kernel_size_and_stride(patch_size, 1)
        self.mrSize = mrSize;
        self.num_ang_bins = 36
        self.patch_size = patch_size;
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), padding = (0,1), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))
        
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), padding = (1,0), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))
        
        self.angular_smooth =  nn.Conv1d(1, 1, kernel_size=3, padding = 1, bias = False)
        self.angular_smooth.weight.data = torch.from_numpy(np.array([[[0.33, 0.34, 0.33]]], dtype=np.float32))
        
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=patch_size).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        if use_cuda:
            self.gk = self.gk.cuda()
        return
    def forward(self, x):
        gx = self.gx(x)
        gy = self.gy(x)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        mag = mag * self.gk.unsqueeze(0).unsqueeze(0).expand_as(mag)
        ori = torch.atan2(gy,gx)
        #mag  = mag * self.gk.expand_as(mag)
        o_big = float(self.num_ang_bins) *(ori + 1.0 * math.pi )/ (2.0 * math.pi)
        bo0_big =  torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big =  bo0_big %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(F.adaptive_avg_pool2d((bo0_big == i).float() * wo0_big, (1,1))) 
        #+ (bo1_big == i).float() * wo1_big))
        ang_bins = torch.cat(ang_bins,1).view(-1,1,self.num_ang_bins)
        ang_bins = self.angular_smooth(ang_bins)
        #print ang_bins[0,:]
        #print ang_bins
        values, indices = ang_bins.view(-1,self.num_ang_bins).max(1)
        return (2. * math.pi * indices.float() / float(self.num_ang_bins)) - 1.0 * math.pi

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
        self.NMS2d = NMS2d(threshold = 0, use_cuda = use_cuda)
        self.OriDet = OrientationDetector(patch_size = 19);
        self.AffShape = AffineShapeEstimator()
        return
    def generate_scale_pyramid(self, x):
        nScales = 5
        sigmaStep = 2 ** (1. / float(nScales))
        pixelDistance = 1.0;
        minSize = 2 * self.b + 2;
        curSigma = 0.5
        #curSigma = 1.6
        if self.init_sigma > curSigma:
            sigma = np.sqrt(self.init_sigma**2 - curSigma**2 + 1e-8)
            curSigma = self.init_sigma
            curr = GaussianBlur(sigma = sigma)(x)
        else:
            sigma = curSigma
            curr = x
        sigmas = [[curSigma]]
        pixel_dists = [[1.0]]
        pyr = [[curr]]
        while True:
            for i in range(1,5):
                sigma = curSigma * np.sqrt(sigmaStep*sigmaStep - 1.0 + 1e-8)
                curr = GaussianBlur(sigma = sigma)(curr)
                curSigma *= sigmaStep
                pyr[-1].append(curr)
                sigmas[-1].append(curSigma)
                pixel_dists[-1].append(pixelDistance)
            pixelDistance = pixelDistance * 2.0
            curr = F.avg_pool2d(curr, kernel_size = 1, stride = 2, padding = 0) 
            curSigma = self.init_sigma
            if (curr[0,0,:,:].size(0) <= minSize) or (curr[0,0,:,:].size(1) <= minSize):
                break
            pyr.append([curr])
            sigmas.append([curSigma])
            pixel_dists.append([pixelDistance])
        return pyr, sigmas, pixel_dists
    def ApplyAffine(self, LAFs, a,b,c):
        A1_ell = torch.cat([a, b], dim = 2)
        A2_ell = torch.cat([b, c], dim = 2)
        A_ell = torch.cat([A1_ell, A2_ell], dim = 1)
        temp_A = torch.bmm(A_ell,LAFs[:,:,0:2])
        return temp_A#torch.cat([temp_A, LAFs[:,:,2:]], dim = 2)
    def rotateLAFs(self, LAFs, angles):
        cos_a = torch.cos(angles).view(-1, 1, 1)
        sin_a = torch.sin(angles).view(-1, 1, 1)
        A1_ang = torch.cat([cos_a, sin_a], dim = 2)
        A2_ang = torch.cat([-sin_a, cos_a], dim = 2)
        A_ang = torch.cat([A1_ang, A2_ang], dim = 1)
        temp_A = torch.bmm(LAFs[:,:,0:2], A_ang )
        return torch.cat([temp_A, LAFs[:,:,2:]], dim = 2)
    def extract_patches(self, scale_pyramid, LAFs, pyr_idxs, PS = 19):
        patches_list = []
        for i in range(len(scale_pyramid)):
            cur_idxs = torch.nonzero((pyr_idxs == i).data)
            if len(cur_idxs.size()) == 0:
                continue
            curr_aff = LAFs[cur_idxs.view(-1), :,:]
            grid = torch.nn.functional.affine_grid(curr_aff, torch.Size((cur_idxs.size(0),
                                                               1,
                                                               PS, 
                                                               PS)))
            patches_list.append(torch.nn.functional.grid_sample(scale_pyramid[i][0].expand(curr_aff.size(0),
                                                                          scale_pyramid[i][0].size(1), 
                                                                          scale_pyramid[i][0].size(2), 
                                                                          scale_pyramid[i][0].size(3)),  grid))
        patches = torch.cat(patches_list, dim = 0)
        return patches
    def forward(self,x):
        ### Generate scale space
        scale_pyr, sigmas, pix_dists = self.generate_scale_pyramid(x)
        ### Detect keypoints in scale space
        aff_matrices = []
        top_responces = []
        pyr_idxs = []
        for oct_idx in range(len(sigmas)):
            print oct_idx
            octave = scale_pyr[oct_idx]
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            for level_idx in range(1,len(octave)-1):
                low = float(sigmas_oct[level_idx - 1 ]**4) * self.Hes(octave[level_idx - 1])
                cur = float(sigmas_oct[level_idx]**4) * self.Hes(octave[level_idx])
                high = float(sigmas_oct[level_idx + 1 ]**4) * self.Hes(octave[level_idx + 1])
                
                nms_f = NMS3dAndComposeA(scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                         mrSize = self.mrSize,
                                        border = self.b)
                top_resp, aff_matrix = nms_f(low,cur,high, self.num / 2)
                #print  np.sum(np.isnan(top_resp.data.cpu().numpy())), np.sum(np.isnan(aff_matrix.data.cpu().numpy()))
                aff_matrices.append(aff_matrix), top_responces.append(top_resp)
                pyr_idxs.append(Variable(oct_idx * torch.ones(aff_matrix.size(0))))
        top_resp_scales = torch.cat(top_responces, dim = 0)
        aff_m_scales = torch.cat(aff_matrices,dim = 0)
        pyr_idxs_scales = torch.cat(pyr_idxs,dim = 0)
        #print top_resp_scales
        final_resp, idxs = torch.topk(top_resp_scales, k = max(1, min(self.num, top_resp_scales.size(0))));
        final_aff_m = torch.index_select(aff_m_scales, 0, idxs)
        final_pyr_idxs = torch.index_select(pyr_idxs_scales,0,idxs)
        patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, PS = 19);
        
        base_A = Variable(torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2))
        if self.use_cuda:
            base_A = base_A.cuda()
        ### Estimate affine shape
        for i in range(self.num_Baum_iters):
            #print i
            a,b,c,eig_ratio = self.AffShape(patches_small)
            base_A = self.ApplyAffine(base_A, a,b,c)
            temp_final = torch.cat([torch.bmm(base_A,final_aff_m[:,:,:2]), final_aff_m[:,:,2:] ], dim =2)
            patches_small = self.extract_patches(scale_pyr, temp_final, final_pyr_idxs, PS = 19)      
        if self.num_Baum_iters > 0:
            final_aff_m = temp_final
        ### Detect orientation
        for i in range(0):
            ori = self.OriDet(patches_small)
            #print np.degrees(ori.data.cpu().numpy().ravel()[1])
            #print final_aff_m[1,:,:]
            #print '*****'
            final_aff_m = self.rotateLAFs(final_aff_m, ori)
            #print final_aff_m[0,:,:]
            patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, PS = 19)
        ### 
        patches = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, PS = self.PS)
        return final_aff_m,patches,final_resp,scale_pyr
    
HA = HessianAffinePatchExtractor( mrSize = 3.0, num_features = 3500, border = 5, num_Baum_iters = 0)
aff, patches, resp, pyr = HA(var_image_reshape / 255.)
LAFs = aff.data.cpu().numpy()
'''
n_patches = patches.size(0)

descriptors_for_net = np.zeros((n_patches, 128))

bs = 128;
outs = []
n_batches = n_patches / bs + 1
t = time.time()
patches = patches.cuda()
hardnet = hardnet.cuda()
for batch_idx in range(n_batches):
    if batch_idx == n_batches - 1:
        if (batch_idx + 1) * bs > n_patches:
            end = n_patches
        else:
            end = (batch_idx + 1) * bs
    else:
        end = (batch_idx + 1) * bs
    data_a = patches[batch_idx * bs: end, :, :, :]
    out_a = hardnet(data_a)
    descriptors_for_net[batch_idx * bs: end,:] = out_a.data.cpu().numpy().reshape(-1, 128)
print descriptors_for_net.shape
et  = time.time() - t
print 'processing', et, et/float(n_patches), ' per patch'
ells = convert_LAF_to_this_stupid_Oxford_ellipse_format(var_image_reshape.data.cpu().numpy()[0,0,:,:], LAFs)

np.savetxt(output_fname, np.hstack([ells, descriptors_for_net]), delimiter=' ', fmt='%10.9f')
'''
ells = convert_LAF_to_this_stupid_Oxford_ellipse_format(var_image_reshape.data.cpu().numpy()[0,0,:,:], LAFs)
np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
line_prepender(output_fname, str(len(ells)))
line_prepender(output_fname, '1.0')
