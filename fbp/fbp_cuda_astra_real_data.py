#-*-coding:utf-8_*_
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
import astra

# 1. load sinogram
sinogram = Image.open('sinogram_remove_line.tiff')
sinogram = np.array(sinogram, dtype=np.float)
plt.figure(), plt.imshow(sinogram, cmap='gray')

# 2. creat geometries
# proj_geom = astra.create_proj_geom('fanflat', 0.13, sinogram.shape[1], np.linspace(0, np.pi, sinogram.shape[0]), 132, 148)
proj_geom = astra.create_proj_geom('parallel', 0.25, sinogram.shape[1], np.linspace(0,  np.pi, sinogram.shape[0]))
vol_geom = astra.create_vol_geom(512, 512)

# 3. creat data id
recon_id = astra.data2d.create('-vol', vol_geom)
proj_id = astra.data2d.create('-sino', proj_geom, sinogram)

projector_id = astra.create_projector('line', proj_geom, vol_geom) 

# 4. config and creat algorithm
cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = recon_id
cfg['ProjectionDataId'] = proj_id
cfg['ProjectorId'] = projector_id
alg_id = astra.algorithm.create(cfg)

# 5. run algorithm
astra.algorithm.run(alg_id, 100)

# 6. get reconstructed volume
rec = astra.data2d.get(recon_id)

# 7. clean up
astra.data2d.delete(recon_id)
astra.data2d.delete(proj_id)
astra.algorithm.delete(alg_id)
astra.data2d.delete(projector_id)

# 8. show the result
plt.figure('rec'), plt.imshow(rec, cmap='gray')
plt.show()
