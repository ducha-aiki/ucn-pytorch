import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from Utils import GaussianBlur, batch_eig2x2, line_prepender
from LAF import LAFs2ell,abc2A, angles2A, generate_patch_grid_from_normalized_LAFs, extract_patches, get_inverted_pyr_index, denormalizeLAFs, extract_patches_from_pyramid_with_inv_index
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid
from NMS import NMS2d, NMS3dAndComposeA

class ScaleSpaceAffinePatchExtractor(nn.Module):
    def __init__(self, use_cuda = False, 
                 border = 16,
                 num_features = 500,
                 patch_size = 32,
                 mrSize = 3.0,
                 nlevels = 8,
                 num_Baum_iters = 0,
                 init_sigma = 1.6,
                 RespNet = None, OriNet = None, AffNet = None):
        super(ScaleSpaceAffinePatchExtractor, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border;
        self.num = num_features
        self.use_cuda = use_cuda
        self.nlevels = nlevels
        self.num_Baum_iters = num_Baum_iters
        self.init_sigma = init_sigma
        if RespNet is not None:
            self.RespNet = RespNet
        else:
            self.RespNet = HessianResp()
        if OriNet is not None:
            self.OriNet = OriNet
        else:
            self.OriNet= OrientationDetector(patch_size = 19);
        if AffNet is not None:
            self.AffNet = AffNet
        else:
            self.AffNet = AffineShapeEstimator(patch_size = 19)
        self.ScalePyrGen = ScalePyramid(nScales = self.nlevels, init_sigma = self.init_sigma, border = self.b)
        return
    def multiScaleDetector(self,x):
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
                low =  self.RespNet(octave[level_idx - 1], (sigmas_oct[level_idx - 1 ]))
                cur = self.RespNet(octave[level_idx ], (sigmas_oct[level_idx ]))
                high = self.RespNet(octave[level_idx + 1], (sigmas_oct[level_idx + 1 ]))
                nms_f = NMS3dAndComposeA(scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                        border = self.b, mrSize = self.mrSize)
                top_resp, aff_matrix = nms_f(low,cur,high, self.num)
                if top_resp is None:
                    break
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
        final_resp, idxs = torch.topk(top_resp_scales, k = max(1, min(self.num, top_resp_scales.size(0))));
        LAFs = torch.index_select(aff_m_scales, 0, idxs)
        final_pyr_idxs = pyr_idxs_scales[idxs]
        final_level_idxs = level_idxs_scale[idxs]
        return final_resp, LAFs, final_pyr_idxs, final_level_idxs,scale_pyr
    def getAffineShape(self,scale_pyr, final_resp, LAFs, final_pyr_idxs, final_level_idxs ):
        pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        #LAFs[:,0:2,0:2] =  LAFs[:,0:2,0:2] / self.init_sigma
        patches_small = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.AffNet.PS, use_cuda = self.use_cuda)
        ###
        base_A = Variable(torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2))
        if self.use_cuda:
            base_A = base_A.cuda()
        ### Estimate affine shape
        for i in range(self.num_Baum_iters):
            #print i
            A = self.AffNet(patches_small)
            base_A = torch.bmm(A, base_A);           
            temp_LAFs = torch.cat([torch.bmm(base_A,LAFs[:,:,0:2]), LAFs[:,:,2:] ], dim =2)
            if i != self.num_Baum_iters - 1:
                patches_small =  extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, temp_LAFs, PS = self.AffNet.PS, use_cuda = self.use_cuda)
            else:
                l1,l2 = batch_eig2x2(base_A)
                ratio = torch.abs(l1 / (l2 + 1e-8))
                idxs_mask = (ratio <= 6.0) * (ratio >= 1./6.)
                idxs_mask = torch.nonzero(idxs_mask.data).view(-1)
                temp_LAFs = temp_LAFs[idxs_mask, :, :]
                final_resp = final_resp[idxs_mask]
                final_pyr_idxs = final_pyr_idxs[idxs_mask]
                final_level_idxs = final_level_idxs[idxs_mask]
        return final_resp, temp_LAFs, final_pyr_idxs, final_level_idxs  
    def getOrientation(self,scale_pyr, LAFs, final_pyr_idxs, final_level_idxs):
        pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        patches_small =  extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS, use_cuda = self.use_cuda)
        max_iters = 0
        ### Detect orientation
        for i in range(max_iters):
            ori = self.OriNet(patches_small)
            #print np.degrees(ori.data.cpu().numpy().ravel()[1])
            LAFs = self.rotateLAFs(LAFs, ori)
            if i != max_iters:
                patches_small = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS, use_cuda = self.use_cuda)        
        return LAFs
    def forward(self,x):
        ### Detection
        final_resp, LAFs, final_pyr_idxs, final_level_idxs,scale_pyr = self.multiScaleDetector(x)
        final_resp, LAFs, final_pyr_idxs, final_level_idxs  = self.getAffineShape(scale_pyr, final_resp, LAFs, final_pyr_idxs, final_level_idxs)
        LAFs[:,:,0:2] = self.mrSize * LAFs[:,:,0:2]
        LAFs = self.getOrientation(scale_pyr, LAFs, final_pyr_idxs, final_level_idxs)
        
        
        pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        patches = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.PS, use_cuda = self.use_cuda)
        return LAFs,patches,final_resp,scale_pyr