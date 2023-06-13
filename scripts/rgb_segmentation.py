import numpy as np
import math
import yaml
import cv2


class RGBSegmentation:
  def __init__(self, cfg_dir):

    self.cfg_dir = cfg_dir
    self.cfg = self.load_configuration()

  def load_configuration(self):
    '''Loads'''
    cfg = None
    with open(self.cfg_dir, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if cfg == None:
       print("Cannot load cfg file to RGB Segmentation class!")

    return cfg
  
  def crop(self, img):
    if len(img.shape) == 3:
        return img[self.cfg['crop']['y1']:self.cfg['crop']['y2'], self.cfg['crop']['x1']:self.cfg['crop']['x2'], :]
    return img[self.cfg['crop']['y1']:self.cfg['crop']['y2'], self.cfg['crop']['x1']:self.cfg['crop']['x2']]
  
  def find_centroid(self, ctr):
    '''finds centroid of the contours'''
    centroid = (0, 0)
    moments = cv2.moments(ctr)
    if moments['m00'] == 0:
        valid = False
    else:
        centroidX = int(moments['m10'] / moments['m00'])
        centroidY = int(moments['m01'] / moments['m00'])
        centroid = (centroidX, centroidY)
        valid = True
    return centroid, valid
  
  def create_cvxhull(self, contours):
    '''Builds convex hull from the contour'''
    mergedContour = np.vstack(contours)
    mergedContour = np.array(mergedContour).reshape((-1, 1, 2)).astype(np.int32)
    cvxhull = cv2.convexHull(mergedContour)
    return cvxhull
  
  def pre_process(self, img):
    '''cleans the image'''
    img_copy = img.copy()
    img_grey = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -5)
    img_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_denoise = cv2.dilate(img_thresh, img_kernel, iterations=3)
    img_denoise = cv2.erode(img_denoise, img_kernel, iterations=3)
    return img_denoise
  
  def filter_contours(self, contours, image_width, image_height):
    contourList = []
    filteredContours = []
    maxArea = 0

    # Go through all contours, recording their information in a list
    # Format of contour list: (index, [points], area of bounding rectangle, coordinates of centroid)
    # Also notes the data for the largest contour in the list
    for number, ctr in enumerate(contours):
        if len(contours) == 1:
            return [contours[0]]

        x, y, w, h = cv2.boundingRect(ctr)

        # Reject contours that are too long, that are probably lines that are not part of an object
        # e.g. edges of the conveyor
        # if w > (0.8 * image_width) or h > (0.8 * image_height):
        #     # this will reject the contour if it is too close to the edges and gets merged into the conveyor contours.
        #     continue

        area = w * h
        centroid, valid = self.find_centroid(ctr)

        if area > maxArea:
            maxArea = area
            largestContour = (number, ctr, area, centroid)

        contourList.append((number, ctr, area, centroid))

    (x, y), radius = cv2.minEnclosingCircle(largestContour[1])

    # Filtering the contours
    for number, ctr, area, centroid in contourList:
        x, y, w, h = cv2.boundingRect(ctr)

        # Reject contours that don't meet minimum size requirements
        if w < self.cfg['rgb']['contour_min_width'] or h < self.cfg['rgb']['contour_min_height']:
            continue

        # Calculate distance metric
        distanceToLargest = math.dist(centroid, largestContour[3])
        if (distanceToLargest / (2 * radius)) > 1:
            continue  # rejects contours that are too far away from the largest contours
        else:
            distanceMetric = (distanceToLargest / (2 * radius)) * 100  # 0 to 100 as distance increases from largest

        # Calculate area metric
        areaMetric = (area / largestContour[2]) * 100
        areaMetric = 100 - areaMetric  # increases as area decreases

        rejectionScore = (self.cfg['rgb']['area_weight'] * areaMetric) + (self.cfg['rgb']['distance_weight'] * distanceMetric)
        if rejectionScore < self.cfg['rgb']['fail_score']:
            filteredContours.append(ctr)
    return filteredContours
  
  def segment(self, img):

    # crop image
    img_cropped = self.crop(img)

    # clean image
    img_clean = self.pre_process(img_cropped)

    # cluster contours in image
    contours, hierarchy = cv2.findContours(img_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_width, image_height, _ = img.shape
    goodContours = self.filter_contours(contours, image_width, image_height)

    try:
        cvxhull = self.create_cvxhull(goodContours)
    except Exception as e:
        raise Exception(f"No object detected: {e}")
    
    area = cv2.contourArea(cvxhull)

    # segmented = img_cropped.copy()
    # x, y, width, height = cv2.boundingRect(cvxhull)
    # cv2.rectangle(segmented, (x, y), (x + width, y + height), (255, 255, 0), 1)
    # cv2.drawContours(segmented, [cvxhull], -1, (0, 0, 255), 1)


    # Create black and white image where contours are filled in white
    img_mask = np.zeros(img_clean.shape, np.uint8)
    cv2.drawContours(img_mask, [cvxhull], -1, (255, 255, 255), thickness=cv2.FILLED)
    mask = img_mask == 255

    return img_cropped, mask, area, cvxhull
  


