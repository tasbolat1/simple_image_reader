import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

data_dir = "../data"
MAX_DEPTH = 600
MAX_COUNT = 5

# Define constants for filtering contours
CONTOUR_MIN_WIDTH = 10
CONTOUR_MIN_HEIGHT = 10
AREA_WEIGHT = 0.35
DISTANCE_WEIGHT = 0.65
FAIL_SCORE = 55


def read_image(obj, count, type='rgb', reduce="mean", n=5):
    # return cv2.imread(f'{data_dir}/{obj}_{count}_{type}.png')
    res = 0
    for i in range(n):
        res += np.load(f'{data_dir}/{obj}_{count}_{type}_{i}.npy')

    if type == 'rgb':
        return res
    
    if reduce == "mean":
        res = res / n

    return res


def PreProcess(image):
    imageCopy = image
    grey = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    denoise = cv2.dilate(thresh, kernel, iterations=3)
    denoise = cv2.erode(denoise, kernel, iterations=3)
    return denoise


def FindCentroid(ctr):
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


def FindConvexHull(contours):
    mergedContour = np.vstack(contours)

    # Reshape into format required by convexHull()
    mergedContour = np.array(mergedContour).reshape((-1, 1, 2)).astype(np.int32)
    hull = cv2.convexHull(mergedContour)
    return hull

def filter_by_height(img, depth_height, subs=0):
    img[img >= MAX_DEPTH] = subs
    return img

def crop(img,x1,x2,y1,y2):
    if len(img.shape) == 3:
        return img[y1:y2, x1:x2, :]
    return img[y1:y2, x1:x2]

def remove_background(img):
    import skimage.exposure

    # # sharpen image
    # kernel = np.array([[0, -1, 0],
    #                     [-1, 3, -1],
    #                     [0, -1, 0]])
    # img = cv2.filter2D(img, -1, kernel)

    # convert to LAB
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

    # extract A channel
    A = lab[:,:,1]

    # threshold A channel
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # blur threshold image
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = skimage.exposure.rescale_intensity(blur, in_range=(175.5,255), out_range=(0,175)).astype(np.uint8)

    # add mask to image as alpha channel
    result = img.copy()
    result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    result[:,:,3] = mask

    return result

def FilterContours(contours, image_width, image_height):
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
        centroid, valid = FindCentroid(ctr)

        if area > maxArea:
            maxArea = area
            largestContour = (number, ctr, area, centroid)

        contourList.append((number, ctr, area, centroid))

    (x, y), radius = cv2.minEnclosingCircle(largestContour[1])

    # Filtering the contours
    for number, ctr, area, centroid in contourList:
        x, y, w, h = cv2.boundingRect(ctr)

        # Reject contours that don't meet minimum size requirements
        if w < CONTOUR_MIN_WIDTH or h < CONTOUR_MIN_HEIGHT:
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

        rejectionScore = (AREA_WEIGHT * areaMetric) + (DISTANCE_WEIGHT * distanceMetric)
        if rejectionScore < FAIL_SCORE:
            filteredContours.append(ctr)
    return filteredContours

def segment_image(img):
    imageCopy = PreProcess(img)
    contours, hierarchy = cv2.findContours(imageCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_width, image_height, _ = img.shape
    goodContours = FilterContours(contours, image_width, image_height)

    try:
        hull = FindConvexHull(goodContours)
    except Exception as e:
        raise Exception(f"No object detected: {e}")
    
    area = cv2.contourArea(hull)

    # Draw point at segment centroid
    # cv2.circle(image, (centroidX, centroidY), 3, (0, 0, 255), 6)
    segmented = img
    # x, y, width, height = cv2.boundingRect(hull)
    # cv2.rectangle(segmented, (x, y), (x + width, y + height), (255, 255, 0), 1)
    # cv2.drawContours(segmented, [hull], -1, (0, 0, 255), 1)

    # Create black and white image where contours are filled in white
    maskImage = np.zeros(imageCopy.shape, np.uint8)
    cv2.drawContours(maskImage, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
    mask = maskImage == 255

    return img, mask, area, segmented
    


obj = 'listerine'
count = 5
img_rgb = read_image(obj, count, type='rgb', n=1)
img_depth = read_image(obj, count, type='depth', n=MAX_COUNT)

# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(img_rgb)
# ax[1].imshow(img_depth)
# plt.show()

# filter by height
img_depth = filter_by_height(img_depth, MAX_DEPTH)
img_depth_cropped = crop(img_depth, x1=110,x2=640,y1=60,y2=390)
img_rgb_cropped = crop(img_rgb, x1=160,x2=640,y1=60,y2=390)
img_rgb_clean = remove_background(img_rgb_cropped)
img_rgb_pre = PreProcess(img_rgb_cropped)
img_rgb, mask, area, img_segmented = segment_image(img_rgb_cropped)

print(mask)

# fig, ax = plt.subplots(ncols=5)
# ax[0].imshow(img_rgb)
# ax[1].imshow(img_depth)
# ax[2].imshow(img_depth_cropped)
# ax[3].imshow(img_rgb_cropped)
# ax[4].imshow(img_rgb_clean)
# plt.show()


fig, ax = plt.subplots(ncols=3)
# ax[0].imshow(img_rgb)
# ax[1].imshow(img_depth)
# ax[2].imshow(img_depth_cropped)
ax[0].imshow(img_rgb_cropped)
ax[1].imshow(img_rgb_pre)
ax[2].imshow(mask)
plt.show()
