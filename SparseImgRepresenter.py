import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy
from Utils import GaussianBlur, batch_eig2x2, line_prepender
from LAF import LAFs2ell,abc2A, angles2A, generate_patch_grid_from_normalized_LAFs, extract_patches, get_inverted_pyr_index, denormalizeLAFs, extract_patches_from_pyramid_with_inv_index, rectifyAffineTransformationUpIsUp
from HandCraftedModules import HessianResp, AffineShapeEstimator, OrientationDetector, ScalePyramid
from NMS import NMS2d, NMS3dAndComposeA

class ScaleSpaceAffinePatchExtractor(nn.Module):
    def __init__(self, 
                 border = 16,
                 num_features = 500,
                 patch_size = 32,
                 mrSize = 3.0,
                 nlevels = 3,
                 num_Baum_iters = 0,
                 init_sigma = 1.6,
                 RespNet = None, OriNet = None, AffNet = None):
        super(ScaleSpaceAffinePatchExtractor, self).__init__()
        self.mrSize = mrSize
        self.PS = patch_size
        self.b = border;
        self.num = num_features
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
        self.ScalePyrGen = ScalePyramid(nLevels = self.nlevels, init_sigma = self.init_sigma, border = self.b)
        return
    
    def multiScaleDetector(self,x, num_features = 0):
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
            low = None
            cur = None
            high = None
            octaveMap = (scale_pyr[oct_idx][0] * 0).byte()
            for level_idx in range(1, len(octave)-1):
                low = self.RespNet(octave[level_idx - 1], (sigmas_oct[level_idx - 1 ]))
                cur = self.RespNet(octave[level_idx ], (sigmas_oct[level_idx ]))
                high = self.RespNet(octave[level_idx + 1], (sigmas_oct[level_idx + 1 ]))
                
                nms_f = NMS3dAndComposeA(scales = sigmas_oct[level_idx - 1:level_idx + 2],
                                        border = self.b, mrSize = self.mrSize)
                top_resp, aff_matrix, octaveMap_current  = nms_f(low, cur, high, octaveMap, num_features = num_features)
                if top_resp is None:
                    continue
                octaveMap = octaveMap_current
                aff_matrices.append(aff_matrix), top_responces.append(top_resp)
                pyr_id = Variable(oct_idx * torch.ones(aff_matrix.size(0)))
                lev_id = Variable((level_idx - 1) * torch.ones(aff_matrix.size(0))) #prevBlur
                if x.is_cuda:
                    pyr_id = pyr_id.cuda()
                    lev_id = lev_id.cuda()
                pyr_idxs.append(pyr_id)
                level_idxs.append(lev_id)
        all_responses = torch.cat(top_responces, dim = 0)
        aff_m_scales = torch.cat(aff_matrices,dim = 0)
        pyr_idxs_scales = torch.cat(pyr_idxs,dim = 0)
        level_idxs_scale = torch.cat(level_idxs, dim = 0)
        if (num_features > 0) and (num_features < all_responses.size(0)):
            all_responses, idxs = torch.topk(all_responses, k = num_features);
            LAFs = torch.index_select(aff_m_scales, 0, idxs)
            final_pyr_idxs = pyr_idxs_scales[idxs]
            final_level_idxs = level_idxs_scale[idxs]
        else:
            return all_responses, aff_m_scales, pyr_idxs_scales , level_idxs_scale, scale_pyr
        return all_responses, LAFs, final_pyr_idxs, final_level_idxs, scale_pyr
    
    def getAffineShape(self,scale_pyr, final_resp, LAFs, final_pyr_idxs, final_level_idxs, num_features = 0, n_iters = 1):
        pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        patches_small = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.AffNet.PS)
        base_A = torch.eye(2).unsqueeze(0).expand(final_pyr_idxs.size(0),2,2)
        if final_resp.is_cuda:
            base_A = base_A.cuda()
        base_A = Variable(base_A)
        is_good = None
        for i in range(n_iters):
            A, is_good_current = self.AffNet(patches_small)
            if is_good is None:
                is_good = is_good_current
            else:
                is_good = is_good * is_good_current
            base_A = torch.bmm(A, base_A); 
            new_LAFs = torch.cat([torch.bmm(base_A,LAFs[:,:,0:2]), LAFs[:,:,2:] ], dim =2)
            if i != self.num_Baum_iters - 1:
                patches_small =  extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, new_LAFs, PS = self.AffNet.PS)
                l1,l2 = batch_eig2x2(A)      
                ratio1 =  torch.abs(l1 / (l2 + 1e-8))
                converged_mask = (ratio1 <= 1.2) * (ratio1 >= (0.8)) 
        l1,l2 = batch_eig2x2(base_A)
        #print l1,l2
        ratio = torch.abs(l1 / (l2 + 1e-8))
        #print new_LAFs[0:2,:,:]
        #print '***'
        #print ((ratio < 6.0) * (ratio > (1./6.))).float().sum()
        #print converged_mask.float().sum()
        #print is_good.float().sum()

        ratio = 1.0 + 0 * torch.abs(l1 / (l2 + 1e-8)) #CHANGE after training
        #idxs_mask = (ratio < 6.0) * (ratio > (1./6.)) * (is_good > 0.5)#  * converged_mask
        idxs_mask = ((ratio < 6.0) * (ratio > (1./6.)))# * converged_mask.float()) > 0
        num_survived = idxs_mask.float().sum()
        #print num_survived
        if (num_features > 0) and (num_survived.data[0] > num_features):
            final_resp =  final_resp * idxs_mask.float() #zero bad points
            final_resp, idxs = torch.topk(final_resp, k = num_features);
        else:
            idxs = torch.nonzero(idxs_mask.data).view(-1).long()
            if (len(idxs.size()) == 0) or (idxs.size(0) == idxs_mask.size(0)):
                idxs = None
        if idxs is not None:
            final_resp = torch.index_select(final_resp, 0, idxs)
            final_pyr_idxs = final_pyr_idxs[idxs]
            final_level_idxs = final_level_idxs[idxs]
            base_A = torch.index_select(base_A, 0, idxs)
            LAFs = torch.index_select(LAFs, 0, idxs)
        #new_LAFs = torch.cat([torch.bmm(rectifyAffineTransformationUpIsUp(base_A), LAFs[:,:,0:2]),
        #                       LAFs[:,:,2:]], dim =2)
        new_LAFs = torch.cat([torch.bmm(base_A, LAFs[:,:,0:2]),
                               LAFs[:,:,2:]], dim =2)
        return final_resp, new_LAFs, final_pyr_idxs, final_level_idxs  
    
    def getOrientation(self,scale_pyr, LAFs, final_pyr_idxs, final_level_idxs):
        pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        patches_small =  extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS)
        max_iters = 1
        ### Detect orientation
        for i in range(max_iters):
            angles = self.OriNet(patches_small)
            #print np.degrees(ori.data.cpu().numpy().ravel()[1])
            LAFs = torch.cat([torch.bmm(angles2A(angles), LAFs[:,:,:2]), LAFs[:,:,2:]], dim = 2)
            if i != max_iters:
                patches_small = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.OriNet.PS)        
        return LAFs
    
    def forward(self,x, random_Baum = False, random_resp = False, return_patches = False):
        ### Detection
        num_features_prefilter = self.num
        #if self.num_Baum_iters > 0:
        #    num_features_prefilter = 2 * self.num;
        if random_resp:
            num_features_prefilter *= 4
        responses, LAFs, final_pyr_idxs, final_level_idxs, scale_pyr = self.multiScaleDetector(x,num_features_prefilter)
        if random_resp:
            if self.num < responses.size(0):
                ridxs = torch.randperm(responses.size(0))[:self.num]
                if x.is_cuda:
                    ridxs = ridxs.cuda() 
                responses = responses[ridxs]
                LAFs = LAFs[ridxs ,:,:]
                final_pyr_idxs = final_pyr_idxs[ridxs]
                final_level_idxs = final_level_idxs[ridxs]
        LAFs[:,0:2,0:2] =   self.mrSize * LAFs[:,:,0:2]
        n_iters = self.num_Baum_iters;
        if random_Baum and (n_iters > 1):
            n_iters = int(np.random.randint(1,n_iters + 1)) 
        if n_iters > 0:
            responses, LAFs, final_pyr_idxs, final_level_idxs  = self.getAffineShape(scale_pyr, responses, LAFs,
                                                                                     final_pyr_idxs, final_level_idxs, self.num, n_iters = n_iters)
        #LAFs = self.getOrientation(scale_pyr, LAFs, final_pyr_idxs, final_level_idxs)
        #if return_patches:
        #    pyr_inv_idxs = get_inverted_pyr_index(scale_pyr, final_pyr_idxs, final_level_idxs)
        #    patches = extract_patches_from_pyramid_with_inv_index(scale_pyr, pyr_inv_idxs, LAFs, PS = self.PS)
        if return_patches:
            patches = extract_patches(x, LAFs, PS = self.PS)
        else:
            patches = None
        return denormalizeLAFs(LAFs, x.size(3), x.size(2)), patches, responses#, scale_pyr
