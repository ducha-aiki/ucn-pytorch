import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import CircularGaussKernel, generate_2dgrid, generate_2dgrid, generate_3dgrid, zero_response_at_border
 
class NMS2d(nn.Module):
    def __init__(self, kernel_size = 3, threshold = 0, use_cuda = False):
        super(NMS2d, self).__init__()
        self.MP = nn.MaxPool2d(kernel_size, stride=1, return_indices=False, padding = 1)
        self.eps = 1e-5
        self.th = threshold
        self.use_cuda = use_cuda
        return
    def forward(self, x):
        local_maxima = self.MP(x)
        if self.th > self.eps:
            return  x * (x > self.th).float() * ((x + self.eps - local_maxima) > 0).float()
        else:
            return ((x - local_maxima + self.eps) > 0).float() * x
        

class NMS3dAndComposeA(nn.Module):
    def __init__(self, mrSize = 1.0, kernel_size = 3, threshold = 0, use_cuda = False, scales = None, border = 3):
        super(NMS3dAndComposeA, self).__init__()
        self.mrSize = mrSize;
        self.eps = 1e-7
        self.ks = 3
        if type(scales) is not list:
            self.grid = generate_3dgrid(3,self.ks,self.ks)
        else:
            self.grid = generate_3dgrid(scales,self.ks,self.ks)
        self.grid = Variable(self.grid.t().contiguous().view(3,3,3,3), requires_grad=False)
        self.th = threshold
        self.use_cuda = use_cuda
        self.cube_idxs = []
        self.border = border
        self.beta = 1.0
        self.grid_ones = Variable(torch.ones(3,3,3,3), requires_grad=False)
        self.NMS2d = NMS2d(kernel_size, threshold, use_cuda )
        if self.use_cuda:
            self.grid = self.grid.cuda()
            self.grid_ones = self.grid_ones.cuda()
        return
    def forward(self, low, cur, high, num_feats = 500):
        assert low.size() == cur.size() == high.size()
        spatial_grid = Variable(generate_2dgrid(low.size(2), low.size(3), False)).view(1,low.size(2), low.size(3),2)
        spatial_grid = spatial_grid.permute(3,1, 2, 0)
        if self.use_cuda:
            spatial_grid = spatial_grid.cuda()
        resp3d = torch.cat([low,cur,high], dim = 1)
        
        #residual_to_patch_center
        softargmax3d = F.conv2d(resp3d,
                                self.grid,
                                padding = 1) / (F.conv2d(resp3d, self.grid_ones, padding = 1) + 1e-8)
        
        #maxima coords
        #print softargmax3d[:,1:,:,:].shape, spatial_grid.shape
        softargmax3d[0,1:,:,:] = softargmax3d[0,1:,:,:] + spatial_grid[:,:,:,0]
        sc_y_x = softargmax3d.view(3,-1).t()
        
        nmsed_resp = zero_response_at_border(self.NMS2d(cur) * ((cur > low) * (cur > high)).float(), self.border)
        
        nmsed_resp_flat = nmsed_resp.view(-1)
        topk_val, idxs = torch.topk(nmsed_resp_flat, 
                                    k = max(1, min(int(num_feats), nmsed_resp_flat.size(0))));
        
        sc_y_x_topk = sc_y_x[idxs.data,:]
        
        sc_y_x_topk[:,1] = sc_y_x_topk[:,1] / float(cur.size(2))
        sc_y_x_topk[:,2] = sc_y_x_topk[:,2] / float(cur.size(3))
        
        min_size = float(min((cur.size(2)), cur.size(3)))
        base_A = Variable(self.mrSize * torch.eye(2).unsqueeze(0).expand(idxs.size(0),2,2).float() / min_size, requires_grad=False)
        if self.use_cuda:
            base_A = base_A.cuda()
        A = sc_y_x_topk[:,:1].unsqueeze(1).expand_as(base_A) * base_A
        full_A  = torch.cat([A,
                             torch.cat([sc_y_x_topk[:,2:].unsqueeze(-1),
                                        sc_y_x_topk[:,1:2].unsqueeze(-1)], dim=1)], dim = 2)
        return topk_val, full_A