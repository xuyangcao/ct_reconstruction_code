#-*-coding:utf-8_*_
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import astra
import phantoms as ph

img = ph.phantom(256)
img = img[::-1]
print img.shape

# 0. parameter settings of fanflat geometry
detector_num = img.shape[0] * 2
views = 540
detector_size = 1
source_origion = 800
origion_det = 200


# 1. creat geometries
proj_geom = astra.create_proj_geom('fanflat', detector_size, detector_num, np.linspace(0, 2 * np.pi, views, False), source_origion, origion_det)
vol_geom = astra.create_vol_geom(img.shape[0], img.shape[0])

# 2. use OpTomo to get W and then sinogram and image
projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
W = astra.OpTomo(projector_id)
print W.shape
sinogram = W * img
haha = W.T * sinogram
haha = np.reshape(haha, (img.shape[0], img.shape[0]))
sinogram = np.reshape(sinogram, (views, detector_num))

# sino = sinogram
# plt.figure('phamtom'), plt.imshow(img, cmap=plt.cm.gray)
# plt.figure('sinogram'), plt.imshow(sino, cmap=plt.cm.gray)

# plt.figure('slice')
# plt.plot(sino[50, :], label='50')
# plt.plot(sino[100, :], label='100')
# plt.plot(sino[150, :], label='150')
# plt.plot(sino[200, :], label='200')
# plt.plot(sino[250, :], label='250')   
# plt.plot(sino[300, :], label='300')
# plt.plot(sino[350, :], label='350')
# plt.plot(sino[500, :], label='500')
# plt.legend()
# plt.show()

# 3. creat data id
recon_id = astra.data2d.create('-vol', vol_geom)
proj_id = astra.data2d.create('-sino', proj_geom, sinogram)

# 4. config and creat algorithm
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = recon_id
cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(cfg)

# 5. run algorithm
astra.algorithm.run(alg_id)

# 6. get reconstructed volume
volume = astra.data2d.get(recon_id)

# 7. show the result
plt.figure('ginogram'), plt.imshow(sinogram, cmap='gray')
plt.figure('rec'), plt.imshow(volume, cmap='gray')
plt.figure('bp'), plt.imshow(haha, cmap='gray')
plt.figure('origin'), plt.imshow(img, cmap='gray')
# display center of image
volume_center = volume[volume.shape[0] / 2, :]
origin_center = img[img.shape[0]/2, :]
plt.figure('center'), plt.plot(volume_center, 'r', label='reconstruct'), plt.plot(origin_center, 'g--', label='origin')
plt.legend()
plt.show()

# 8. clean up
astra.data2d.delete(recon_id)
astra.data2d.delete(proj_id)
astra.algorithm.delete(alg_id)
astra.data2d.delete(projector_id)