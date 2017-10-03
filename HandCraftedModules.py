import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from Utils import CircularGaussKernel, GaussianBlur
from LAF import abc2A,rectifyAffineTransformationUpIsUp

class ScalePyramid(nn.Module):
    def __init__(self, nLevels = 3, init_sigma = 1.6, border = 5, use_cuda = False):
        super(ScalePyramid,self).__init__()
        self.nLevels = nLevels;
        self.init_sigma = init_sigma
        self.sigmaStep =  2 ** (1. / float(self.nLevels))
        #print 'step',self.sigmaStep
        self.b = border
        self.minSize = 2 * self.b + 2 + 1;
        self.use_cuda = use_cuda;
        return
    def forward(self,x):
        pixelDistance = 1.0;
        curSigma = 0.5
        if self.init_sigma > curSigma:
            sigma = np.sqrt(self.init_sigma**2 - curSigma**2)
            curSigma = self.init_sigma
            curr = GaussianBlur(sigma = sigma, use_cuda = self.use_cuda)(x)
        else:
            curr = x
        sigmas = [[curSigma]]
        pixel_dists = [[1.0]]
        pyr = [[curr]]
        while True:
            curr = pyr[-1][0]
            for i in range(1, self.nLevels + 2):
                sigma = curSigma * np.sqrt(self.sigmaStep*self.sigmaStep - 1.0 )
                #print 'blur sigma', sigma
                curr = GaussianBlur(sigma = sigma, use_cuda = self.use_cuda)(curr)
                curSigma *= self.sigmaStep
                pyr[-1].append(curr)
                sigmas[-1].append(curSigma)
                pixel_dists[-1].append(pixelDistance)
                if i == self.nLevels:
                    nextOctaveFirstLevel = F.avg_pool2d(curr, kernel_size = 1, stride = 2, padding = 0) 
            pixelDistance = pixelDistance * 2.0
            curSigma = self.init_sigma
            if (nextOctaveFirstLevel[0,0,:,:].size(0)  <= self.minSize) or (nextOctaveFirstLevel[0,0,:,:].size(1) <= self.minSize):
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append([curSigma])
            pixel_dists.append([pixelDistance])
        return pyr, sigmas, pixel_dists

class HessianResp(nn.Module):
    def __init__(self):
        super(HessianResp, self).__init__()
        
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))

        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))

        self.gxx =  nn.Conv2d(1, 1, kernel_size=(1,3),bias = False)
        self.gxx.weight.data = torch.from_numpy(np.array([[[[1.0, -2.0, 1.0]]]], dtype=np.float32))
        
        self.gyy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gyy.weight.data = torch.from_numpy(np.array([[[[1.0], [-2.0], [1.0]]]], dtype=np.float32))
        return
    def forward(self, x, scale):
        gxx = self.gxx(F.pad(x, (1,1,0, 0), 'replicate'))
        gyy = self.gyy(F.pad(x, (0,0, 1,1), 'replicate'))
        gxy = self.gy(F.pad(self.gx(F.pad(x, (1,1,0, 0), 'replicate')), (0,0, 1,1), 'replicate'))
        return torch.abs(gxx * gyy - gxy * gxy) * (scale**4)


class AffineShapeEstimator(nn.Module):
    def __init__(self, use_cuda = False, 
                 threshold = 0.001, patch_size = 19):
        super(AffineShapeEstimator, self).__init__()
        self.threshold = threshold;
        self.use_cuda = use_cuda;
        self.PS = patch_size
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3), bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=patch_size, circ_zeros = False).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        if use_cuda:
            self.gk = self.gk.cuda()
        return
    def invSqrt(self,a,b,c):
        eps = 1e-12
        mask = (b != 0).float()
        r1 = mask * (c - a) / (2. * b + eps)
        t1 = torch.sign(r1) / (torch.abs(r1) + torch.sqrt(1. + r1*r1));
        r = 1.0 / torch.sqrt( 1. + t1*t1)
        t = t1*r;
        
        r = r * mask + 1.0 * (1.0 - mask);
        t = t * mask;
        
        x = 1. / torch.sqrt( r*r*a - 2*r*t*b + t*t*c)
        z = 1. / torch.sqrt( t*t*a + 2*r*t*b + r*r*c)
        
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
        gx = self.gx(F.pad(x, (1,1,0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0,0, 1,1), 'replicate'))
        a1 = (gx*gx * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        b1 = (gx*gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        c1 = (gy*gy * self.gk.unsqueeze(0).unsqueeze(0).expand_as(gx)).view(x.size(0),-1).mean(dim=1)
        l1, l2, a, b, c = self.invSqrt(a1,b1,c1)
        rat1 = l1/l2
        mask = (torch.abs(rat1) <= 6.).float().view(-1);
        a = a * mask + 1. * (1.- mask)
        b = b * mask + 0. * (1.- mask)
        c = c * mask + 1. * (1.- mask)
        return torch.cat([torch.cat([a.unsqueeze(-1).unsqueeze(-1), b.unsqueeze(-1).unsqueeze(-1)], dim = 2),
                                        torch.cat([b.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)], dim = 2)],
                                        dim = 1), mask
        

class OrientationDetector(nn.Module):
    def __init__(self, use_cuda = False,
                mrSize = 3.0, patch_size = None):
        super(OrientationDetector, self).__init__()
        if patch_size is None:
            patch_size = 32;
        self.PS = patch_size;
        self.bin_weight_kernel_size, self.bin_weight_stride = self.get_bin_weight_kernel_size_and_stride(self.PS, 1)
        self.mrSize = mrSize;
        self.num_ang_bins = 36
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3),  bias = False)
        self.gx.weight.data = torch.from_numpy(np.array([[[[0.5, 0, -0.5]]]], dtype=np.float32))
        
        self.gy =  nn.Conv2d(1, 1, kernel_size=(3,1), bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[0.5], [0], [-0.5]]]], dtype=np.float32))
        
        self.angular_smooth =  nn.Conv1d(1, 1, kernel_size=3, padding = 1, bias = False)
        self.angular_smooth.weight.data = torch.from_numpy(np.array([[[0.33, 0.34, 0.33]]], dtype=np.float32))
        
        self.gk = torch.from_numpy(CircularGaussKernel(kernlen=self.PS).astype(np.float32))
        self.gk = Variable(self.gk, requires_grad=False)
        if use_cuda:
            self.gk = self.gk.cuda()
        return
    def get_bin_weight_kernel_size_and_stride(self, patch_size, num_spatial_bins):
        bin_weight_stride = int(round(2.0 * np.floor(patch_size / 2) / float(num_spatial_bins + 1)))
        bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
        return bin_weight_kernel_size, bin_weight_stride

    def forward(self, x):
        gx = self.gx(F.pad(x, (1,1,0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0,0, 1,1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        mag = mag * self.gk.unsqueeze(0).unsqueeze(0).expand_as(mag)
        ori = torch.atan2(gy,gx)
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
        return (2. * float(np.pi) * indices.float() / float(self.num_ang_bins)) - 1.0 * float(np.pi)
