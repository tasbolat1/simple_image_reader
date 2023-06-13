import numpy as np
import matplotlib.pyplot as plt
from rgb_segmentation import RGBSegmentation
from depth_segmentation import DEPTHSegmentation
from utils import *

data_dir = "../data"
MAX_COUNT = 5

def read_image(obj, count, type='rgb', reduce="mean", n=5):
    # return cv2.imread(f'{data_dir}/{obj}_{count}_{type}.png')
    res = 0
    for i in range(n):
        res += np.load(f'{data_dir}/{obj}_{count}_{type}_{i}.npy')

    if type == 'rgb':
        return res
    
    if reduce == "mean":
        res = res / n

    if type == "depth":
        res = res / 1000.0

    return res


obj = 'babyoil'
count = 5
objs = ['labo', 'shampoo', 'babyoil', 'listerine']
K = np.array([607.8644409179688, 0.0, 327.0859069824219, 0.0, 608.1785278320312, 241.8994903564453, 0.0, 0.0, 1.0])
K = K.reshape([3,3])

img_rgb = read_image(obj, count, type='rgb', n=1)
img_depth = read_image(obj, count, type='depth', n=MAX_COUNT)
rgb_segmentor = RGBSegmentation(cfg_dir='../conf/vision_conf.yaml')
depth_segmentor = DEPTHSegmentation(cfg_dir='../conf/vision_conf.yaml')
depth_segmentor.set_K(K)

img_cropped, mask, area, cvx_hull = rgb_segmentor.segment(img_rgb)

img_depth_cropped = depth_segmentor.crop(img_depth)
pc, pc_colors = depth_segmentor.depth2pc_with_color(img_depth, img_rgb)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(img_depth)
# ax[1].imshow(img_depth_clean)
plt.show()

import open3d as o3d

pc_o3d = o3d.geometry.PointCloud()  # create point cloud object
pc_o3d.points = o3d.utility.Vector3dVector(pc)  # set pcd_np as the point cloud points
# Visualize:
pc_o3d.colors = o3d.utility.Vector3dVector(pc_colors)
o3d.visualization.draw_geometries([pc_o3d])

# ax[0].imshow(img_rgb)
# ax[1].imshow(img_cropped)
# img_contoured = draw_contour(img_cropped, cvx_hull)
# ax[2].imshow(img_contoured)
# plt.show()

# for obj in objs: 
#     for count in range(1,6):
#         img_rgb = read_image(obj, count, type='rgb', n=1)
#         img_depth = read_image(obj, count, type='depth', n=MAX_COUNT)
#         rgb_segmentor = RGBSegmentation(cfg_dir='../conf/vision_conf.yaml')
#         img_cropped, mask, area, cvx_hull = rgb_segmentor.segment(img_rgb)

#         fig, ax = plt.subplots(ncols=3)
#         ax[0].imshow(img_rgb)
#         ax[1].imshow(img_cropped)
#         a = draw_contour(img_cropped, cvx_hull)
#         ax[2].imshow(a)
#         plt.show()