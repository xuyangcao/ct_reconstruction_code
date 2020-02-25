import astra 
import numpy as np 
import matplotlib.pyplot as plt 
import phantoms as ph 

class AstraToolbox:
    
    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, np.linspace(0,np.pi,n_angles,False))
        self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom)
        
        self.matrix_id = astra.projector.matrix(self.proj_id)
        self.A = astra.matrix.get(self.matrix_id)
        # self.A = self.A.toarray()
            
        self.n_pixels = n_pixels
        self.n_angles = n_angles

    def proj(self, img):
        sino = self.A.dot(img.ravel()).reshape(self.n_angles, -1)
        return sino

    def backproj(self, sino_data):
        rec = (self.A.T).dot(sino_data.ravel()).reshape(self.n_pixels, -1)
        return rec

    def cleanup(self):
        astra.projector.delete(self.proj_id)
        astra.matrix.delete(self.matrix_id)

def power_method(W, sino, width, n_it = 20 ): #algorithm 3
    x = np.ones((width, width), dtype=np.float32)
    for i in range(0, n_it):
        x = AT(A(x))
        x /= np.sqrt(norm2sq(x))
        s = np.sqrt(norm2sq(A(x)))
    return s

def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())
    # return np.sum(mat**2)

def chamble_pock(L, W, g, img, Lambda, n_it, return_cPD=True):
    sigma = 1.0 / L
    tau = 1.0 / L
    theta = 1.0

    u = np.zeros(img.shape, dtype=np.float32)
    p = np.zeros(g.shape, dtype=np.float32)
    u_tilde = np.zeros(u.shape, dtype=np.float32)

    if return_cPD:
        cPD = np.zeros(n_it, dtype=np.float32)
    
    for k in range(n_it):
        p = (p + sigma*(A(u_tilde) - g)) / (1.0+sigma)
        u_old = u
        u = u - tau * AT(p)
        u_tilde = u + theta * (u - u_old)

        cpd = 0.5 * norm2sq(A(u)-g) + 0.5 * norm2sq(p) + np.sum(p*g)
        cPD[k] = abs(cpd)

        if k % 10 == 0:
            print("%d: %e" % (k, cPD[k]))
    if return_cPD:
        return cPD, u
    else:
        return u

if __name__ == '__main__':
    # 1, read image 
    img = ph.phantom(64)
    img = img[::-1]

    # 2, parameter setting 
    n_angles = 160 # number of proj. angles
    Lambda=1
    n_it = 20000
    ast = AstraToolbox(img.shape[0], n_angles)   
    A = lambda x: ast.proj(x)
    AT = lambda y: ast.backproj(y)

    # 3, sinogram
    sino = A(img)
    
    #4. cp
    L = power_method(ast.A, sino, img.shape[0], 30)
    print L
    cPD, rec = chamble_pock(L, ast.A, sino, img, Lambda, n_it)
    plt.figure(), plt.imshow(rec, cmap='gray')
    plt.figure(), plt.plot(cPD)

    # 5. show slice 
    slice_img = img[img.shape[0]/2, :]
    slice_rec = rec[rec.shape[0]/2, :]
    plt.figure()
    plt.plot(slice_img, 'g-', label='original image')
    plt.plot(slice_rec, 'r--', label='recon image')
    plt.legend()

    # 5. clean up
    ast.cleanup()
    plt.show()