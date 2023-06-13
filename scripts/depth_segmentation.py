import numpy as np
import cv2
import yaml

class DEPTHSegmentation:
  def __init__(self, cfg_dir):

    self.cfg_dir = cfg_dir
    self.cfg = self.load_configuration()

    self.depthCleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_64F, self.cfg['depth']['cleaner_window_size'], cv2.rgbd.DepthCleaner_DEPTH_CLEANER_NIL )

  def set_K(self, K):
     self.fx = K[0,0] #607.8644409179688
     self.fy = K[1,1] #608.1785278320312
     self.cx = K[0,2] #327.0859069824219
     self.cy = K[1,2] #241.8994903564453

  def load_configuration(self):
    '''Loads'''
    cfg = None
    with open(self.cfg_dir, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if cfg == None:
       print("Cannot load cfg file to Depth Segmentation class!")

    return cfg
  
  def clean(self, img):
    '''Cleans the image using cv2 default method'''
    img_copy = img.copy()
    output_image = self.depthCleaner.apply(img_copy)    
    return output_image
  
  def crop(self, img):
    if len(img.shape) == 3:
        return img[self.cfg['crop']['y1']:self.cfg['crop']['y2'], self.cfg['crop']['x1']:self.cfg['crop']['x2'], :]
    return img[self.cfg['crop']['y1']:self.cfg['crop']['y2'], self.cfg['crop']['x1']:self.cfg['crop']['x2']]
  
  def depth2pc_with_color(self, img, img_rgb):
    pcd = []
    colors = []
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            z = img[i][j]
            x = (j - self.cx) * z / self.fx
            y = (i - self.cy) * z / self.fy
            pcd.append([x, y, z])
            colors.append(img_rgb[i][j]/255)

    return np.array(pcd), np.array(colors)

  def depth2pc(self, img):

    pcd = []
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            z = img[i][j]
            x = (j - self.cx) * z / self.fx
            y = (i - self.cy) * z / self.fy
            pcd.append([x, y, z])

    return np.array(pcd)

     