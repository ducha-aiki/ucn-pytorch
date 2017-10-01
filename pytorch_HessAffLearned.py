import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy


from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell, abc2A


LOG_DIR = 'log_snaps'
BASE_LR = 0.00000001
USE_CUDA = False


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


class BaumNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, PS = 16):
        super(BaumNet, self).__init__()
        self.PS = PS
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
        return abc2A(abc[:,0,:,:].contiguous() + 1. ,abc[:,1,:,:].contiguous() , abc[:,2,:,:].contiguous() + 1.)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal(m.weight.data, gain=1.0)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return

HA = ScaleSpaceAffinePatchExtractor( mrSize = 5.0, num_features = 3000, border = 1, num_Baum_iters = 5, AffNet = BaumNet())

LAFs, patches, resp, pyr = HA(var_image_reshape / 255.)
LAFs = denormalizeLAFs(LAFs, img.shape[1], img.shape[0], use_cuda = USE_CUDA).data.cpu().numpy()

ells = LAFs2ell(LAFs)

np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
#line_prepender(output_fname, str(len(ells)))
#line_prepender(output_fname, '1.0')