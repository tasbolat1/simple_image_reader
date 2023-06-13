import rosbag
import glob
import cv_bridge
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re

objs = ['labo', 'shampoo', 'babyoil', 'listerine']
bag_data_dir = '../data/original_data'
save_data_dir = '../data'
MAX_COUNT = 10

bag_files = glob.glob(f'{bag_data_dir}/*.bag')


bridge = CvBridge()

for fname in bag_files:
    for obj in objs:
        if obj in fname:
            idx = re.search(obj, fname).start() + len(obj) + 1
            count = fname[:idx][-1]
            with rosbag.Bag(fname, 'r') as bag:
                print('Original')
                for i, (topic, msg, ts) in enumerate( bag.read_messages(topics=str('/camera/color/camera_info')) ):
                    print(msg.K)
                    break
                print('Depth')
                for i, (topic, msg, ts) in enumerate( bag.read_messages(topics=str('/camera/aligned_depth_to_color/camera_info')) ):
                    print(msg.K)
                    break


                # for i, (topic, msg, ts) in enumerate( bag.read_messages(topics=str('/camera/color/image_raw')) ):
                #     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    
                #     # cv2.imwrite(f'{save_data_dir}/{obj}_{count}_rgb.png', cv_image)
                #     with open(f'{save_data_dir}/{obj}_{count}_rgb_{i}.npy', 'wb') as f:
                #         np.save(f, cv_image)
                #     print(obj, count)
                #     if i == MAX_COUNT:
                #         break
                # for i, (topic, msg, ts) in enumerate( bag.read_messages(topics=str('/camera/aligned_depth_to_color/image_raw')) ):
                #     cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                #     # cv2.imwrite(f'{bag_data_dir}/{obj}_{count}_depth.png', cv_image)
                #     # print(cv_image.shape)
                #     # print(cv_image)
                #     with open(f'{save_data_dir}/{obj}_{count}_depth_{i}.npy', 'wb') as f:
                #         np.save(f, cv_image)
                    
                #     if i == MAX_COUNT:
                #         break
