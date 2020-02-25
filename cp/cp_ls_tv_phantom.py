# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:34:16 2017

@author: new_xuyangcao

this script demostrated cp algorithm to solve least square and TV panalty objective.

"""

import astra 
import numpy as np
import phantoms as ph 
import matplotlib.pyplot as plt 

import numpy as np 
import phantoms as ph 
import matplotlib.pyplot as plt 
import astra 
from math import sqrt 
import time

class AstraToolbox:
    
    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, np.linspace(0,np.pi,n_angles,False))
        self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
        #~ self.rec_id = astra.data2d.create('-vol', self.vol_geom)
        
        self.matrix_id = astra.projector.matrix(self.proj_id)
        self.A = astra.matrix.get(self.matrix_id)
            
        self.n_pixels = n_pixels
        self.n_angles = n_angles

    def proj(self, slice_data):
        sino = (self.A.dot(slice_data.ravel())).reshape(self.n_angles, -1)
        return sino

    def backproj(self, sino_data):
        rec = (self.A.T.dot(sino_data.ravel())).reshape(self.n_pixels, -1)
        # rec /= sqrt(norm2sq(rec))
        return rec

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        # astra.data2d.delete(self.proj_id)
        astra.projector.delete(self.proj_id)
        astra.matrix.delete(self.matrix_id)



def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient

def __gradient(img):
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    gradient[0, :-1, :] = np.diff(img, axis=0)
    gradient[0, -1, :] = -img[-1, :]

    gradient[1, :, :-1] = np.diff(img, axis=1)
    gradient[1, :, -1] = -img[:, -1]

    return gradient


def div(grad):
    '''
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    grad[:, 0, :] = 0
    grad[:, -1, :] = 0
    grad[:, :, 0] = 0
    grad[:, :, -1] = 0

    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d) 
        this_res = np.rollaxis(res, d)          # *this_res and res point to the same address
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res

def norm1(mat):
    return np.sum(np.abs(mat))

def norm2sq(mat):
    # return np.dot(mat.ravel(), mat.ravel())
    return np.sum(mat**2)

def proj_l2(g, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g**2, 0))/Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res

def power_method(W, sino, width, n_it = 10 ): #algorithm 3
    x = np.ones((width, width), dtype=np.float32)
    # x = np.mat(x)
    # x = (A.T * sino).reshape(width, width)
    for i in range(0, n_it):
        x = AT(A(x)) - div(gradient(x))
        x /= sqrt(norm2sq(x))
        s = sqrt(norm2sq(A(x)) + norm2sq(gradient(x)))
    return s

def chamble_pock(L, W, g, img, Lambda, n_it, return_cPD=True):
    sigma = 1.0 / L
    tau = 1.0 / L

    u = np.zeros(img.shape, dtype=np.float64)
    p = np.zeros(g.shape, dtype=np.float64)
    q = np.zeros(gradient(u).shape, dtype=np.float64)
    u_tilde = np.zeros(u.shape, dtype=np.float64)

    theta = 1.0

    if return_cPD: 
        cPD = np.zeros(n_it)
        cc = np.zeros(n_it)
        pg = np.zeros(n_it)

    for k in range(n_it):
        p = (p + sigma*(A(u_tilde) - g)) / (1.0+sigma)
        deno = np.sqrt(np.sum((q + sigma * gradient(u_tilde))**2, axis=0))
        q = Lambda * (q + sigma * gradient(u_tilde)) / np.maximum(Lambda, deno)

        # q = proj_l2(q +sigma*gradient(u_tilde), Lambda)
        u_old = u
        u = u - tau * AT(p) + tau * div(q)
        u_tilde = u + theta * (u - u_old)

        cpd = 0.5 * norm2sq(A(u)-g) + Lambda * np.sum(np.sqrt(np.sum(gradient(u)**2, axis=0))) + 0.5 * norm2sq(p)  + np.dot(p.ravel(), g.ravel())
        cc[k] = 0.5 * norm2sq(A(u)-g) + Lambda * np.sum(np.sqrt(np.sum(gradient(u)**2, axis=0))) + 0.5 * norm2sq(p)
        pg[k] = np.dot(p.ravel(), g.ravel())

        #cpd = 2
        cPD[k] = abs(cpd)
        if np.abs(cpd) < 0:
            print cpd
            break
        if k % 1 == 0:
            print("%d: %f: \t cc: %f \t pg: %f" % (k, cPD[k], cc[k], pg[k]))
    if return_cPD:
        plt.figure(), plt.plot(cc)
        plt.figure(), plt.plot(pg)
        return cPD, u
    else:
        return u

if __name__ == '__main__':
    # 1, read image 
    img = ph.phantom(64)
    img = img[::-1]
    plt.imshow(img, cmap = 'gray'), plt.title('original image')

    # 2, parameter setting 
    n_angles = 60 # number of proj. angles
    Lambda = 2e-4
    n_it = 10000
    ast = AstraToolbox(img.shape[0], n_angles)   
    A = lambda x: ast.proj(x)
    AT = lambda y: ast.backproj(y)
    
    # 2, sinogram
    sino = A(img)
    plt.figure(), plt.imshow(sino, cmap='gray'), plt.title('sino')

    # 3, chamble-pock
    L = power_method(ast.A, sino, img.shape[0], 30)
    print L
    cPD, rec = chamble_pock(L, ast.A, sino, img, Lambda, n_it)
    plt.figure(), plt.imshow(rec, cmap='gray')
    plt.figure(), plt.plot(cPD)

    # 4. show slice 
    slice_img = img[img.shape[0]/2, :]
    slice_rec = rec[rec.shape[0]/2, :]
    plt.figure()
    plt.plot(slice_img, 'g-', label='original image')
    plt.plot(slice_rec, 'r--', label='recon image')
    plt.legend()

    # 5. clean up
    ast.cleanup()
    plt.show()