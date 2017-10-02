import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.linalg import inv    
from scipy.linalg import schur, sqrtm
import torch
from  torch.autograd import Variable

def Ell2LAF(ell):
    A23 = np.zeros((2,3))
    A23[0,2] = ell[0]
    A23[1,2] = ell[1]
    a = ell[2]
    b = ell[3]
    c = ell[4]
    C = np.array([[a, b], [b, c]])
    sc = np.sqrt(a*c - b*b)
    A23[0:2,0:2] = sqrtm(C)[::-1,::-1] / sc
    A23[1,0] = -A23[1,0]
    A23[0,1] = 0
    return A23


def ells2LAFs(ells):
    LAFs = np.zeros((len(ells), 2,3))
    for i in range(len(ells)):
        LAFs[i,:,:] = Ell2LAF(ells[i,:])
    return LAFs

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
    H_pts_out = np.transpose(np.matmul(HLAF, H_pts))
    H_pts_out[:,0] = H_pts_out[:,0] / H_pts_out[:, 2]
    H_pts_out[:,1] = H_pts_out[:,1] / H_pts_out[:, 2]
    return H_pts_out[:,0:2]

def abc2A(a,b,c, normalize = False):
    A1_ell = torch.cat([a.view(-1,1,1), b.view(-1,1,1)], dim = 2)
    A2_ell = torch.cat([b.view(-1,1,1), c.view(-1,1,1)], dim = 2)
    return torch.cat([A1_ell, A2_ell], dim = 1)

def rectifyAffineTransformationUpIsUp(A):
    det = torch.sqrt(torch.abs(A[:,0,0]*A[:,1,1] - A[:,1,0]*A[:,0,1] + 1e-10))
    b2a2 = torch.sqrt(A[:,0,1] * A[:,0,1] + A[:,0,0] * A[:,0,0])
    A1_ell = torch.cat([(b2a2 / det).contiguous().view(-1,1,1), 0 * A[:,1,1].contiguous().view(-1,1,1)], dim = 2)
    A2_ell = torch.cat([((A[:,1,1]*A[:,0,1]+A[:,1,0]*A[:,0,0])/(b2a2*det)).contiguous().view(-1,1,1), (det / b2a2).contiguous().view(-1,1,1)], dim = 2)
    return torch.cat([A1_ell, A2_ell], dim = 1)

def angles2A(angles):
    cos_a = torch.cos(angles).view(-1, 1, 1)
    sin_a = torch.sin(angles).view(-1, 1, 1)
    A1_ang = torch.cat([cos_a, sin_a], dim = 2)
    A2_ang = torch.cat([-sin_a, cos_a], dim = 2)
    return  torch.cat([A1_ang, A2_ang], dim = 1)

def generate_patch_grid_from_normalized_LAFs(LAFs, w, h, PS, use_cuda = False):
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3) * 0.5  * min_size
    coef[0,0,2] = w
    coef[0,1,2] = h
    if use_cuda:
        coef = coef.cuda()
    coef = Variable(coef.expand(num_lafs,2,3))
    grid = torch.nn.functional.affine_grid(LAFs * coef, torch.Size((num_lafs,1,PS,PS)))
    grid[:,:,:,0] = 2.0 * grid[:,:,:,0] / float(w)  - 1.0
    grid[:,:,:,1] = 2.0 * grid[:,:,:,1] / float(h)  - 1.0     
    return grid
    
def extract_patches(img, LAFs, PS = 32, use_cuda = False):
    w = img.size(3)
    h = img.size(2)
    ch = img.size(1)
    grid = generate_patch_grid_from_normalized_LAFs(LAFs, float(w),float(h), PS, use_cuda )
    return torch.nn.functional.grid_sample(img.expand(grid.size(0), ch, h, w),  grid)  

def extract_patches_from_pyramid_with_inv_index(scale_pyramid, pyr_inv_idxs, LAFs, PS = 19, use_cuda = False):
    patches = torch.zeros(LAFs.size(0),scale_pyramid[0][0].size(1), PS, PS)
    if use_cuda:
        patches = patches.cuda()
    patches = Variable(patches)
    if pyr_inv_idxs is not None:
        for i in range(len(scale_pyramid)):
            for j in range(len(scale_pyramid[i])):
                cur_lvl_idxs = pyr_inv_idxs[i][j]
                if cur_lvl_idxs is None:
                    continue
                cur_lvl_idxs = cur_lvl_idxs.view(-1)
                patches[cur_lvl_idxs,:,:,:] = extract_patches(scale_pyramid[i][j],LAFs[cur_lvl_idxs, :,:], PS, use_cuda )
    return patches

def get_inverted_pyr_index(scale_pyr, pyr_idxs, level_idxs):
    pyr_inv_idxs = []
    ### Precompute octave inverted indexes
    for i in range(len(scale_pyr)):
        pyr_inv_idxs.append([])
        cur_idxs = pyr_idxs == i #torch.nonzero((pyr_idxs == i).data)
        for j in range(0, len(level_idxs)):
            cur_lvl_idxs = torch.nonzero(((level_idxs == j) * cur_idxs).data)
            if len(cur_lvl_idxs.size()) == 0:
                pyr_inv_idxs[-1].append(None)
            else:
                pyr_inv_idxs[-1].append(cur_lvl_idxs.squeeze(1))
    return pyr_inv_idxs


def denormalizeLAFs(LAFs, w, h, use_cuda = False):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3)  * min_size
    coef[0,0,2] = w
    coef[0,1,2] = h
    if use_cuda:
        coef = coef.cuda()
    coef = Variable(coef.expand(num_lafs,2,3))
    return coef * LAFs

def normalizeLAFs(LAFs, w, h, use_cuda = False):
    w = float(w)
    h = float(h)
    num_lafs = LAFs.size(0)
    min_size = min(h,w)
    coef = torch.ones(1,2,3).float()  / min_size
    coef[0,0,2] = 1.0 / w
    coef[0,1,2] = 1.0 / h
    if use_cuda:
        coef = coef.cuda()
    coef = Variable(coef.expand(num_lafs,2,3))
    return coef * LAFs

    
def convertLAFs_to_A23format(LAFs):
    sh = LAFs.shape
    if (len(sh) == 3) and (sh[1]  == 2) and (sh[2] == 3): # n x 2 x 3 classical [A, (x;y)] matrix
        work_LAFs = deepcopy(LAFs)
    elif (len(sh) == 2) and (sh[1]  == 7): #flat format, x y scale a11 a12 a21 a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] * LAFs[:,3] 
        work_LAFs[:,0,1] = LAFs[:,2] * LAFs[:,4]
        work_LAFs[:,1,0] = LAFs[:,2] * LAFs[:,5]
        work_LAFs[:,1,1] = LAFs[:,2] * LAFs[:,6]
    elif (len(sh) == 2) and (sh[1]  == 6): #flat format, x y s*a11 s*a12 s*a21 s*a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] 
        work_LAFs[:,0,1] = LAFs[:,3]
        work_LAFs[:,1,0] = LAFs[:,4]
        work_LAFs[:,1,1] = LAFs[:,5]
    else:
        print 'Unknown LAF format'
        return None
    return work_LAFs

def LAFs2ell(in_LAFs):
    LAFs = convertLAFs_to_A23format(in_LAFs)
    ellipses = np.zeros((len(LAFs),5))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        scale = np.sqrt(LAF[0,0]*LAF[1,1]  - LAF[0,1]*LAF[1, 0] + 1e-10)
        u, W, v = np.linalg.svd(LAF[0:2,0:2] / scale, full_matrices=True)
        W[0] = 1. / (W[0]*W[0]*scale*scale)
        W[1] = 1. / (W[1]*W[1]*scale*scale)
        A =  np.matmul(np.matmul(u, np.diag(W)), u.transpose())
        ellipses[i,0] = LAF[0,2]
        ellipses[i,1] = LAF[1,2]
        ellipses[i,2] = A[0,0]
        ellipses[i,3] = A[0,1]
        ellipses[i,4] = A[1,1]
    return ellipses

def visualize_LAFs(img, LAFs):
    work_LAFs = convertLAFs_to_A23format(LAFs)
    plt.figure()
    plt.imshow(255 - img)
    for i in range(len(work_LAFs)):
        ell = LAF2pts(work_LAFs[i,:,:])
        plt.plot( ell[:,0], ell[:,1], 'r')
    plt.show()
    return 