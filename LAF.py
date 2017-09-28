import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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

def visualize_LAFs(img, LAFs):
    plt.figure()
    plt.imshow(255 - img)
    min_shape = min(float(img.shape[1]),float(img.shape[0]))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        LAF[:,:2] *= min_shape
        LAF[0,2] *= float(img.shape[1])
        LAF[1,2] *= float(img.shape[0])
        #print LAF
        ell = LAF2pts(LAF)
        plt.plot( ell[:,0], ell[:,1], 'r')
    plt.show()
    return

def LAFs2ell(img, LAFs):
    h,w = img.shape
    min_shape = min(h,w)
    ellipses = np.zeros((len(LAFs),5))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        LAF[0,2] *= float(img.shape[1])
        LAF[1,2] *= float(img.shape[0])
        scale = np.sqrt(LAF[0,0]*LAF[1,1]  - LAF[0,1]*LAF[1, 0] + 1e-10)
        LAF[0:2,0:2] /=  scale;
        scale *= float(min_shape)
        u, W, v = np.linalg.svd(LAF[0:2,0:2], full_matrices=True)
        W[0] = 1. / (W[0]*W[0]*scale*scale)
        W[1] = 1. / (W[1]*W[1]*scale*scale)
        A =  np.matmul(np.matmul(u, np.diag(W)), u.transpose())
        ellipses[i,0] = LAF[0,2]
        ellipses[i,1] = LAF[1,2]
        ellipses[i,2] = A[0,0]
        ellipses[i,3] = A[0,1]
        ellipses[i,4] = A[1,1]
    return ellipses
