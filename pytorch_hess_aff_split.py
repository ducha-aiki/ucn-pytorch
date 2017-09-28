import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

from PIL import Image
from pytorch_sift import SIFTNet
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy


from Utils import GaussianBlur, batch_eig2x2, line_prepender
from LAF import LAFs2ell
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid
from NMS import NMS2d, NMS3dAndComposeA


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
        self.OriDet = OrientationDetector(patch_size = 19);
        self.AffShape = AffineShapeEstimator(patch_size = 19)
        self.ScalePyrGen = ScalePyramid(nScales = self.nlevels, init_sigma = self.init_sigma, border = self.b)
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
    def extract_patches(self, scale_pyramid, LAFs, pyr_idxs, level_idxs, PS = 19, gauss_mask = False, use_cuda = False):
        patches_list = []
        if gauss_mask:
            mask = torch.from_numpy(CircularGaussKernel(kernlen = PS, circ_zeros = False).astype(np.float32))
            mask = Variable(mask)
            if use_cuda:
                mask = mask.cuda()
        for i in range(len(scale_pyramid)):
            cur_idxs = pyr_idxs == i #torch.nonzero((pyr_idxs == i).data)
            for j in range(1, len(level_idxs) - 1):
                cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
                if len(cur_lvl_idxs.size()) == 0:
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
            print oct_idx
            octave = scale_pyr[oct_idx]
            sigmas_oct = sigmas[oct_idx]
            pix_dists_oct = pix_dists[oct_idx]
            for level_idx in range(1,len(octave)-1):
                low = float(sigmas_oct[level_idx - 1 ]**4) * self.Hes(octave[level_idx - 1])
                cur = float(sigmas_oct[level_idx]**4) * self.Hes(octave[level_idx])
                high = float(sigmas_oct[level_idx + 1 ]**4) * self.Hes(octave[level_idx + 1])
                nms_f = NMS3dAndComposeA(scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                         mrSize = 1.0,
                                        border = self.b)
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
        #final_aff_m[:,:,0:2] =  final_aff_m[:,:,0:2] / self.init_sigma
        patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs,final_level_idxs, PS = 19, gauss_mask = False);
        ###
        
        base_A = Variable(torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2))
        if self.use_cuda:
            base_A = base_A.cuda()
        ### Estimate affine shape
        for i in range(self.num_Baum_iters):
            print i
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
                patches_small = self.extract_patches(scale_pyr, temp_final, final_pyr_idxs, final_level_idxs, PS = 19, gauss_mask = False)
            else:
                idxs_mask = torch.nonzero(((ratio <= 6.0) * (ratio >= 1./6.)).data).view(-1)
                temp_final = temp_final[idxs_mask, :, :]
                final_pyr_idxs = final_pyr_idxs[idxs_mask]
                final_level_idxs = final_level_idxs[idxs_mask]
            
        #
        if self.num_Baum_iters > 0:
            final_aff_m = temp_final
        #####
        #final_aff_m[:,:,0:2] = self.init_sigma * self.mrSize * final_aff_m[:,:,0:2]
        final_aff_m[:,:,0:2] =  self.mrSize * final_aff_m[:,:,0:2]
        patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs, PS = 19, gauss_mask = False)
        ######
        ### Detect orientation
        for i in range(0):
            ori = self.OriDet(patches_small)
            #print np.degrees(ori.data.cpu().numpy().ravel()[1])
            #print final_aff_m[1,:,:]
            #print '*****'
            final_aff_m = self.rotateLAFs(final_aff_m, ori)
            #print final_aff_m[0,:,:]
            patches_small = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs,  PS = 19, gauss_mask = False)
        ###
        patches = self.extract_patches(scale_pyr, final_aff_m, final_pyr_idxs, final_level_idxs, PS = self.PS)
        return final_aff_m,patches,final_resp,scale_pyr
    
HA = HessianAffinePatchExtractor( mrSize = 5.196, num_features = 4000, border = 5, num_Baum_iters = 16)
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
ells = LAFs2ell(var_image_reshape.data.cpu().numpy()[0,0,:,:], LAFs)
np.savetxt(output_fname, ells, delimiter=' ', fmt='%10.10f')
line_prepender(output_fname, str(len(ells)))
line_prepender(output_fname, '1.0')