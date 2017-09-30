import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.linalg import inv    
from scipy.linalg import schur, sqrtm

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