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
from LAF import denormalizeLAFs, LAFs2ell
from Utils import line_prepender


LOG_DIR = 'log_snaps'
BASE_LR = 0.00000001
USE_CUDA = False


try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
    nfeats = int(sys.argv[3])
except:
    print "Wrong input format. Try ./extract_hardnet_desc_from_hpatches_file.py imgs/ref.png out.txt 2000"
    sys.exit(1)

img = Image.open(input_img_fname).convert('RGB')
img = np.mean(np.array(img), axis = 2)

var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)))
var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))

    
HA = ScaleSpaceAffinePatchExtractor( mrSize = 1.0, num_features = nfeats, border = 5, num_Baum_iters = 0)

LAFs, patches, resp, pyr = HA(var_image_reshape / 255.)
LAFs = denormalizeLAFs(LAFs, img.shape[1], img.shape[0], use_cuda = USE_CUDA).data.cpu().numpy()

ells = LAFs2ell(LAFs)

np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
#line_prepender(output_fname, str(len(ells)))
#line_prepender(output_fname, '1.0')