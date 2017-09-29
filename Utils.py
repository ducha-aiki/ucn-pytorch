import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np

# resize image to size 32x32
cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32),
                                 interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def CircularGaussKernel(kernlen=None, circ_zeros = False, sigma = None, norm = True):
    assert ((kernlen is not None) or sigma is not None)
    if kernlen is None:
        kernlen = int(2.0 * 3.0 * sigma + 1.0)
        if (kernlen % 2 == 0):
            kernlen = kernlen + 1;
        halfSize = float(kernlen) / 2.;
    else:
        halfSize = kernlen / 2;
    r2 = float(halfSize*halfSize)
    if sigma is None:
        sigma2 = 0.9 * r2;
        sigma = np.sqrt(sigma2)
    else:
        sigma2 = sigma * sigma    
    x = np.linspace(0,kernlen-1,kernlen)
    xv, yv = np.meshgrid(x, x, sparse=False, indexing='xy')
    distsq = (xv - halfSize)**2 + (yv - halfSize)**2
    kernel = np.exp(-( distsq/ (sigma2)))
    if circ_zeros:
        kernel *= (distsq <= r2).astype(np.float32)
    if norm:
        kernel /= np.sum(kernel)
    return kernel

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

class GaussianBlur(nn.Module):
    def __init__(self, sigma=1.6, use_cuda = False):
        super(GaussianBlur, self).__init__()
        weight = self.calculate_weights(sigma)
        self.register_buffer('buf', weight)
        self.use_cuda = use_cuda
        return
    def calculate_weights(self,  sigma):
        kernel = CircularGaussKernel(sigma = sigma, circ_zeros = False)
        h,w = kernel.shape
        halfSize = float(h) / 2.;
        self.pad = int(np.floor(halfSize))
        return torch.from_numpy(kernel.astype(np.float32)).view(1,1,h,w);
    def forward(self, x):
        w = Variable(self.buf)
        if self.use_cuda:
            w=w.cuda()
        return F.conv2d(x, w, padding = self.pad)

def batch_eig2x2(A):
    trace = A[:,0,0] + A[:,1,1]
    delta1 = (trace*trace - 4 * ( A[:,0,0]*  A[:,1,1] -  A[:,1,0]* A[:,0,1]))
    mask = delta1 > 0
    delta = torch.sqrt(torch.abs(delta1))
    l1 = mask.float() * (trace + delta) / 2.0 +  100.  * (1 - mask.float())
    l2 = mask.float() * (trace - delta) / 2.0 +  0.001  * (1 - mask.float())
    return l1,l2

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    return
