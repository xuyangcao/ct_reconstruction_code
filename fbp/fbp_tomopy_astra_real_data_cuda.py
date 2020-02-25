import tomopy 
import astra
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import bytescale
from skimage.restoration import denoise_nl_means

'''
this scripts shows how to use tomopy with astra using fbp algorithm.

the data structure is a little bit different than that in astra.

so a demo with real data is showed here.

'''
if __name__ == '__main__':

    # 1. load image 
    sino = Image.open('sinogram.tiff')
    sino = np.array(sino, dtype=np.float32)

    # 2. denoise projection data using nlm algorithm 
    # sino = denoise_nl_means(sino, patch_size=3, patch_distance=7, h=0.01)
    plt.figure(), plt.imshow(sino, cmap='gray'), plt.axis('off')

    # 3. convert projection data structure 
    proj = np.zeros((sino.shape[0], 1,  sino.shape[1]))
    proj[:, 0, :] = sino
    print proj.shape

    # 4. views initionalize 
    theta = tomopy.angles(sino.shape[0], 0, 180)

    # 5. find center 
    # center = tomopy.find_center(proj, theta, ind=0, init=1042, tol=0.5)
    # print center

    # 6. using tomopy's FBP API reconstruction 
    # rec = tomopy.recon(proj, theta, center=1046.47734375, algorithm='fbp', filter_name='')#, num_gridx=256, num_gridy=256)
    # print rec.shape

    # 7. using tomopy plus astra 
    options = {'proj_type':'line', 'method':'FBP_CUDA'}
    rec = tomopy.recon(proj, theta, center=1046.47734375, algorithm=tomopy.astra, options=options)

    # 8. convert data structure 
    rec = rec[0, :, :]

    # 9. show reconstruction result
    plt.figure(), plt.imshow(rec, cmap='gray'), plt.axis('off')
    plt.figure(), plt.plot(sino[344, :], 'b')

    plt.show()