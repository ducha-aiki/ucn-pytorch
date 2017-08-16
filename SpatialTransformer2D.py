import torch
import torch.nn as nn
import numpy as np

def crop_center(img,cropx,cropy):
    y,x = img.shape
    if cropx < x:
        startx = x//2-(cropx//2)
        endx = startx+cropx
    else:
        startx = 0
        endx = x
    if cropy < y:
        starty = y//2-(cropy//2)
        endy = starty+cropy
    else:
        starty = 0
        endy = y
    return img[starty:endy,startx:endx]
class SpatialTransformer2d(nn.Module):
    def __init__(self, 
                 num_input_channels = 1,
                 feature_net = None,
                 out_patch_size = 32,
                 out_stride = 8,
                 min_zoom = 1.0,
                 max_zoom = 1.0,
                 min_tilt = 1.0,
                 max_tilt = 1.0,
                 max_rot = 0,
                 max_shift = 0,
                 mrSize = 3.0):
        super(SpatialTransformer2d, self).__init__()
        
        ### geometrical restrictions
        self.in_planes = num_input_channels
        self.out_patch_size = out_patch_size;
        self.max_rot = max_rot;
        self.min_zoom = min_zoom;
        self.min_tilt = min_tilt;
        self.max_zoom = max_zoom;
        self.max_tilt = max_tilt;
        self.max_shift = max_shift;
        self.mrSize = mrSize;
        self.extraction_mrSize = self.max_zoom * (1.0 + self.max_shift) * self.mrSize;
        self.gridPatchSize = int(np.ceil( self.extraction_mrSize * self.out_patch_size))
        ###
        if feature_net is None:
            self.spatial_transformer_feature_net = nn.Sequential(
                nn.Conv2d(num_input_channels, 16, kernel_size=3, padding = 1, bias = False),
                nn.BatchNorm2d(16, affine=False),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias = False),
                nn.BatchNorm2d(32, affine=False),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2,padding=1, bias = False),
                nn.BatchNorm2d(64, affine=False),
                nn.ReLU()
            )
        else:
            self.spatial_transformer_feature_net = feature_net
        self.spatial_transformer_feature_net.cuda()
        stride_h, stride_w = self.get_net_stride()
        assert stride_h == stride_w # I am too lazy to deal with non-square patches
        self.ST_features_stride = stride_h
        assert self.ST_features_stride <= out_stride  #
        assert out_stride % self.ST_features_stride == 0 #Otherwise we are in trouble
        self.last_layer_stride = int(out_stride / self.ST_features_stride);
        
        self.out_patch_size = out_patch_size; 
        self.out_stride = out_stride;# num of strides in ST net

        ### Parameters networks 
        self.psi_net =  nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=self.last_layer_stride, bias = True, stride = self.last_layer_stride),
            nn.Tanh()
        )
        self.theta_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=self.last_layer_stride, bias = True, stride = self.last_layer_stride),
            nn.Tanh()
        )
        self.shift_net = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=self.last_layer_stride, bias = True, stride = self.last_layer_stride),
            nn.Tanh()
        )
        self.iso_scale_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=self.last_layer_stride, bias = True, stride = self.last_layer_stride),
            nn.Tanh()
        )
        self.horizontal_tilt_net = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=self.last_layer_stride, bias = True, stride = self.last_layer_stride),
            nn.Tanh()
        )
        ### Useful constants
        self.eye2 = torch.autograd.Variable(torch.eye(2))
        self.eye2 = self.eye2.cuda()
        self.zero = torch.autograd.Variable(torch.zeros(1,1,1))
        self.zero = self.zero.cuda()
        self.one = torch.autograd.Variable(torch.ones(1,1,1))
        self.one = self.one.cuda()
        return
    def get_net_stride(self):
        #Everything should be zero padded in ST_feature_net, so size is reduced be strides only 
        in_h = in_w = 1024
        inp = torch.autograd.Variable(torch.rand(1, self.in_planes, in_h, in_w)).cuda()
        out_f = self.spatial_transformer_feature_net.forward(inp)
        #print out_f.size();
        stride_h = in_h / out_f.size(2);
        stride_w = in_w / out_f.size(3);
        return stride_h, stride_w
    def get_rotation_matrix(self, angle_in_radians):
        angle_in_radians = angle_in_radians.view(-1, 1, 1);
        sin_a = torch.sin(angle_in_radians)
        cos_a = torch.cos(angle_in_radians)
        A1_x = torch.cat([cos_a, -sin_a], dim = 2)
        A2_x = torch.cat([sin_a, cos_a], dim = 2)
        transform = torch.cat([A1_x,A2_x], dim = 1)
        return transform

    def get_tilt_matrix(self, horizontal_tilt): #X-axis anisoptropic scale
        an_s = horizontal_tilt.view(-1, 1, 1)
        A1_x = torch.cat((an_s, self.zero.expand_as(an_s)), dim = 2)
        A2_x = torch.cat([self.zero.expand_as(an_s), self.one.expand_as(an_s) / an_s], dim = 2)
        return torch.cat([A1_x,A2_x], dim = 1)
    
    def get_scale_matrix(self, iso_scale): #Isotropic scale
        iso_s = iso_scale.view(-1, 1, 1)
        A1_x = torch.cat((iso_s, self.zero.expand_as(iso_s)), dim = 2)
        A2_x = torch.cat([self.zero.expand_as(iso_s), iso_s], dim = 2)
        return torch.cat([A1_x,A2_x], dim = 1)
    
    def compose_affine_matrix(self, psi, theta, iso_scale, horizontal_tilt, shift): 
        ## See illustration in MODS paper for notation
        # Output is n x 2 x 3 Affine transformation matrix 
        in_plane_rot =  self.get_rotation_matrix(psi);
        out_plane_rot = self.get_rotation_matrix(theta);
        tilt_matrix = self.get_tilt_matrix(horizontal_tilt);
        iso_scale = iso_scale.view(-1,1,1)
        A_iso_scale  = self.get_scale_matrix(iso_scale);
        A_iso_scale_in_plane = torch.bmm(A_iso_scale,in_plane_rot);
        A_tilt_out_of_place = torch.bmm(tilt_matrix, out_plane_rot)
        A_no_shift = torch.bmm(A_iso_scale_in_plane, A_tilt_out_of_place)
        return torch.cat([A_no_shift, shift.view(-1,2,1)], dim = 2)

    def unfold_patches_for_grid(self,input_image, grid_size):
        #To avoid zeros, we should sample transformed patch from bigger patch, that final (in case of zoom-out or tilt)
        #needed_y = int(np.ceil(float(input_image.size(2)) / self.out_stride))
        #needed_x = int(np.ceil(float(input_image.size(3)) / self.out_stride))
        needed_y = grid_size[2]
        needed_x = grid_size[3]
        #floor((L_{in}  + 2 * padding - dilation * (kernel\_size - 1) - 1) / stride + 1)`
        needed_y_pad = (needed_y - 1) * self.out_stride + self.gridPatchSize - input_image.size(2)
        needed_x_pad = (needed_x - 1) * self.out_stride + self.gridPatchSize - input_image.size(3)
        padx_2 = int(np.ceil(float(needed_x_pad) / 2))
        pady_2 = int(np.ceil(float(needed_y_pad) / 2))
        half_ps = self.gridPatchSize / 2
        padded = nn.ZeroPad2d((padx_2,needed_x_pad - padx_2, pady_2,needed_y_pad - pady_2))(input_image)
        #print input_image.shape, padded.shape
        inp_unfolded1 = padded.unfold(2, self.gridPatchSize, self.out_stride).squeeze(1);
        #print inp_unfolded1.shape
        inp_unfolded2 = inp_unfolded1.unfold(2,self.gridPatchSize, self.out_stride).squeeze(1);
        #print inp_unfolded2.shape
        
        inp_unfolded = inp_unfolded2.contiguous().view(-1,1,self.gridPatchSize,self.gridPatchSize)
        patch_centers_x = torch.arange(half_ps , input_image.size(3) + padx_2 + 1,
                                       step=self.out_stride) - padx_2;
        patch_centers_y = torch.arange(half_ps , input_image.size(2) + pady_2 + 1,
                                       step=self.out_stride) -  pady_2;
        patch_centers = torch.stack([patch_centers_x.repeat(patch_centers_y.size(0)), 
                                         patch_centers_y.repeat(patch_centers_x.size(0),1).t().contiguous().view(-1)],1)
        patch_centers = torch.autograd.Variable(patch_centers)
        patch_centers = patch_centers.cuda()
        return inp_unfolded, patch_centers
    def fold_image_back_numpy(self, patches, image_width, image_height):
        from  scipy.ndimage import zoom as imzoom
        num,ch,patch_h, patch_w = patches.shape
        num_patches_y = int(np.floor(float(image_height) / self.out_stride));
        num_patches_x = int(np.floor(float(image_width) / self.out_stride));
        out_img = np.zeros((num_patches_y * self.out_stride ,num_patches_x * self.out_stride));
        #print out_img.shape
        #
        ps_in_img = self.out_stride;
        idx = 0
        make_bigger = False
        make_crop = False
        if ps_in_img > patch_w:
            make_bigger = True
            zoom = float(ps_in_img) / patch_w;
        elif ps_in_img < patch_w:
            make_crop = True
        for j in range(num_patches_y):
            for i in range(num_patches_x):
                if make_crop:
                    out_img[ps_in_img*j:ps_in_img*(j+1),ps_in_img*i:ps_in_img*(i+1)] =  crop_center(patches[idx ,0,:,:], ps_in_img,ps_in_img)
                elif make_bigger:
                    out_img[ps_in_img*j:ps_in_img*(j+1),ps_in_img*i:ps_in_img*(i+1)] =  imzoom(patches[idx ,0,:,:],(zoom,zoom))
                else:
                    out_img[ps_in_img*j:ps_in_img*(j+1),ps_in_img*i:ps_in_img*(i+1)] =  patches[idx ,0,:,:]
                idx+=1
        return crop_center(out_img, image_width, image_height)
    def input_norm(self,x):
        #Local patch normalization for descriptors.
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input_img):
        ST_features = self.spatial_transformer_feature_net(input_img)
        psi = self.max_rot * self.psi_net(ST_features)
        theta = self.max_rot * self.theta_net(ST_features)
        shift = self.max_shift * self.shift_net(ST_features) 
        tilt = torch.clamp(1.0 + (self.max_tilt - 1) * self.horizontal_tilt_net(ST_features),
                          min = self.min_tilt, max = self.max_tilt)
        scale = torch.clamp(1.0 + (self.max_zoom - 1) * self.iso_scale_net(ST_features),
                            min = self.min_zoom, max = self.max_zoom)

        transform = self.compose_affine_matrix(psi, theta, scale, tilt, shift)
        #print transform.shape
        #print transform
        grid = torch.nn.functional.affine_grid(transform, torch.Size((transform.size(0),
                                                           self.in_planes,
                                                           self.out_patch_size, 
                                                           self.out_patch_size)))
        #print grid.shape
        inp_unfolded, patch_centers = self.unfold_patches_for_grid(input_img, psi.size());
        
        grid =  grid / float(self.gridPatchSize / self.out_patch_size)
        #adjust grid for taking bigger input patch
        #to avoid empty places. Input patch size is calculates in unfold_patches_for_grid function
        #print inp_unfolded.shape,grid.shape
        transformed_patches = torch.nn.functional.grid_sample(inp_unfolded, grid)
        affine_matrices_out = self.out_patch_size * transform
        #print patch_centers.shape,affine_matrices_out[:,:,2].shape
        affine_matrices_out[:,:,2] =  patch_centers + affine_matrices_out[:,:,2];
        return transformed_patches, affine_matrices_out