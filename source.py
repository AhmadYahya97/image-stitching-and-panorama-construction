import math
import numpy as np
import cv2 as cv2
import glob
import imutils
import os
import time
import matplotlib.pyplot as plt


width = int(500)
height = int(700)

def saveEnhancedImage(image,innerDirectory,name):
    if(not os.path.exists(str(innerDirectory))):
        os.makedirs(str(innerDirectory))
    return cv2.imwrite((str(innerDirectory)+'/'+str(name)+'.jpg'),image)


def rotate2(img,angle,times):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    for i in range(times):
        img = cv2.warpAffine(img, M, (width, height))
    return img
#--------------------------#
def rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """
    angle = math.radians(angle)
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)

def crop(img, w, h):
    x, y = int(img.shape[1] * .5), int(img.shape[0] * .5)

    return img[
        int(np.ceil(y - h * .5)) : int(np.floor(y + h * .5)),
        int(np.ceil(x - w * .5)) : int(np.floor(x + h * .5))
    ]

def rotate(img, angle):
    # rotate, crop and return original size
    (h, w) = img.shape[:2]
    img = imutils.rotate_bound(img, angle)
    img = crop(img, *rotated_rect(w, h, angle))
    img = cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
    return img
#--------------------------#

#--------------------------#
def showAndWait(img):
    x=1
    # cv2.imshow('im', img)
    # cv2.waitKey(0)
#--------------------------#


#--------------------------#
def show(img):
    cv2.imshow('im', img)
#--------------------------#

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#--------------------------#
def getImage(path):
    return cv2.imread(path)
#--------------------------#


#--------------------------#
def resize(img,w,h):
    return  cv2.resize(img, (w, h))
#--------------------------#


#--------------------------#
# def rotate(img,times):
#     M = cv2.getRotationMatrix2D((width / 2, height / 2), 15, 1)
#     for i in range(times):
#         img = cv2.warpAffine(img, M, (width, height))
#     return img
#--------------------------#


#--------------------------#
def returnFeatureDetectionAlgorithm(algoName):
    if(algoName == 'SIFT'):
        return cv2.xfeatures2d.SIFT_create()
    if(algoName == 'SURF'):
        return  cv2.xfeatures2d.SURF_create()
    if(algoName == 'ORB'):
        return cv2.ORB_create(nfeatures=1500)
    return None
#--------------------------#


#--------------------------#
def getKeyPoints_Descriptors(algo,img):
    return algo.detectAndCompute(img, None)
#--------------------------#


#--------------------------#
def returnMatchingImage(img1, kp1, img2, kp2, matches, k):
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:k], None)
#--------------------------#


#--------------------------#
def findMinAbsSlope(algoName,img1, img2, k, knn = True):
    algo = returnFeatureDetectionAlgorithm(algoName)

    kps1, des1 = getKeyPoints_Descriptors(algo, img1)

    kps2, des2 = getKeyPoints_Descriptors(algo, img2)

    matches = returnSortedMatchesBetweenTwoImages(algoName, des1, des2, knn)

    val = 0
    dis = 0
    for i in matches[:k]:
        x1, y1 = kps1[i.queryIdx].pt
        x2, y2 = kps2[i.trainIdx].pt
        if (x2 == x1):
            val = val + 0
        else:
            val = val + abs((y2 - y1) / (x2 - x1))
        dis = dis + ((y2 - y1)**2 + (x2+width - x1)**2)**0.5

    return val*dis
#--------------------------#





#--------------------------#
def findMinimumSlopeSidePair(algoName, img1, img2, k, knn = True):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    mn = None
    side1, side2 = 0, 0
    for i in range(4):
        for j in range(4):
            val = findMinAbsSlope(algoName, img1, img2, k, knn)
            if(mn == None or val < mn):
                side1 = i
                side2 = j
                mn = val
            img2 = cv2.warpAffine(img2, M, (width, height))
        img1 = cv2.warpAffine(img1, M, (width, height))
    return (mn,side1, side2)
#--------------------------#


#--------------------------#
def findMinimumSlopeSidePair2(algoName, img1, img2, k, knn = True):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 1)
    mn = None
    side = 0
    for j in range(4):
        val = findMinAbsSlope(algoName, img1, img2, k, knn)
        if(mn == None or val < mn):
            side = j
            mn = val
        img1 = cv2.warpAffine(img1, M, (width, height))
    return (mn,side)
#--------------------------#


#--------------------------#
def returnSortedMatchesBetweenTwoImages(algoName,des1, des2, knn = True):
    if(algoName == 'SIFT' or algoName == 'SURF'):
        if(knn == True):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                    good.append(m[0])

            return sorted(good, key=lambda x: x.distance)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            return sorted(matches, key=lambda x: x.distance)

    if(algoName == 'ORB'):
        if(knn == True):
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m in matches:
                if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                    good.append(m[0])
            return sorted(good, key=lambda x: x.distance)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
            matches = bf.match(des1, des2)
            return sorted(matches, key = lambda x:x.distance)
#--------------------------#
#--------------------------#
def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
    reprojThresh,matches):

    newMatches = []

    for m in matches:
      newMatches.append((m.trainIdx,m.queryIdx))

    ptsA = np.float32([kpsA[i] for (_, i) in newMatches])
    ptsB = np.float32([kpsB[i] for (i, _) in newMatches])

    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
    return (newMatches, H, status)
#--------------------------#

def getGaussianFilter(image,size,sigma):
    return cv2.GaussianBlur(image,(size, size),sigma)

def illuminate(image,alpha,beta):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
                image[y][x][0] = np.clip(alpha*image[y][x][0] + beta, 0, 255)
                image[y][x][1] = np.clip(alpha * image[y][x][1] + beta, 0, 255)
                image[y][x][2] = np.clip(alpha * image[y][x][2] + beta, 0, 255)

    return image

def addGaussianNoise(image,mean,stdv):
    noise = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    noise = cv2.randn(noise,mean,stdv)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

                image[i][j][0]=int(image[i][j][0])+int(noise[i][j])

                image[i][j][1] = int(image[i][j][1]) + int(noise[i][j])

                image[i][j][2] = int(image[i][j][2]) + int(noise[i][j])
    return image
#--------------------------#
def stitch(algoName, imagesPath, k, knn = True,operation='NONE',blur = 0):

    totalTime =0
    algo = returnFeatureDetectionAlgorithm(algoName)
    image_list = []
    for filename in glob.glob(imagesPath + '/*.*'):
        # print(filename.split(imagesPath + '/')[1])
        # print(filename)
        im = resize(getImage(filename), width, height)
        if(operation=='BLUR'):
         im = getGaussianFilter(im,20,blur)
        if(operation == 'NOISE'):
         im = addGaussianNoise(im,0,angle)
        if(operation == 'GAMMA'):
            im = illuminate(im,10,0)

        image_list.append((im, filename.split(imagesPath + '/')[1].split('.')[0]))
    image_list = sorted(image_list, key=lambda x: x[1])
    for i in range(len(image_list)):
        image_list[i] = image_list[i][0]
    # image_list[0] = rotate(image_list[0],angle)

    # Just for two photos [0] and [1]
    if(len(image_list) >= 2):
        # image_list[len(image_list) - 2] = rotate_im(image_list[len(image_list) - 2],20)
        showAndWait(image_list[len(image_list) - 2])
        showAndWait(image_list[len(image_list) - 1])
        mm, side1, side2 = findMinimumSlopeSidePair('ORB' ,image_list[len(image_list) - 2], image_list[len(image_list) - 1], k, False)
        # image_list[len(image_list) - 2] = rotate(image_list[len(image_list) - 2], side1)
        # image_list[len(image_list) - 1] = rotate(image_list[len(image_list) - 1], side2)
        imageA = image_list[len(image_list) - 1]
        imageB = image_list[len(image_list) - 2]
        start = time.time()
        kps1, des1 = getKeyPoints_Descriptors(algo, imageA)
        kps2, des2 = getKeyPoints_Descriptors(algo, imageB)
        matches = returnSortedMatchesBetweenTwoImages(algoName, des1, des2, knn)
        # imageB = rotate(imageB, kps1[matches[0].queryIdx].angle - kps2[matches[0].trainIdx].angle)
        kps1, des1 = getKeyPoints_Descriptors(algo, imageA)
        kps2, des2 = getKeyPoints_Descriptors(algo, imageB)

        matches = returnSortedMatchesBetweenTwoImages(algoName, des1, des2, knn)


        end = time.time()
        totalTime = totalTime + (end - start)
        m = returnSortedMatchesBetweenTwoImages(algoName, des2, des1, knn)
        mm = returnMatchingImage(imageB, kps2, imageA, kps1, m, k);
        showAndWait(mm)
        saveEnhancedImage(mm,'matches',algoName)
        start = time.time()
        kps1 = np.float32([kp.pt for kp in kps1])
        kps2 = np.float32([kp.pt for kp in kps2])
        M = matchKeypoints(kps1, kps2, des1, des2, 4,matches)
        (matches, H, status) = M
        matchcount = len(matches)

        result = cv2.warpPerspective(imageA, H,
                                     (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        end = time.time()
        totalTime = totalTime + (end - start)

        # result = resize(result, 750, 750)
        # _, thresh = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
        #
        # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        # x, y, w, h = cv2.boundingRect(cnt)
        # result = result[y:y+h - 1,x:x+w - 1]
        for i in range(len(image_list) - 2):
            showAndWait(result)
            showAndWait(image_list[len(image_list) - i - 3])
            mm, side = findMinimumSlopeSidePair2('ORB' ,image_list[len(image_list) - i - 3], result, k, False)
            # image_list[len(image_list) - i - 3] = rotate(image_list[len(image_list) - i - 3], side)
            imageA = result
            imageB = image_list[len(image_list) - i - 3]
            start = time.time()
            kps1, des1 = getKeyPoints_Descriptors(algo, imageA)
            kps2, des2 = getKeyPoints_Descriptors(algo, imageB)
            matches = returnSortedMatchesBetweenTwoImages(algoName, des1, des2, knn)

            # imageB = rotate(imageB, kps1[matches[0].queryIdx].angle - kps2[matches[0].trainIdx].angle)
            kps1, des1 = getKeyPoints_Descriptors(algo, imageA)
            kps2, des2 = getKeyPoints_Descriptors(algo, imageB)

            matches = returnSortedMatchesBetweenTwoImages(algoName, des1, des2, knn)
            end = time.time()
            totalTime = totalTime + (end - start)
            m = returnSortedMatchesBetweenTwoImages(algoName, des2, des1, knn)
            mm = returnMatchingImage(imageB, kps2 , imageA, kps1, m, k);
            showAndWait(mm)
            start = time.time()
            kps1 = np.float32([kp.pt for kp in kps1])
            kps2 = np.float32([kp.pt for kp in kps2])
            M = matchKeypoints(kps1, kps2, des1, des2, 4, matches)
            (matches, H, status) = M
            result = cv2.warpPerspective(imageA, H,
                                         (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            end = time.time()
            totalTime = totalTime + (end - start)
            result = resize(result, width, height)
            # _, thresh = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
            # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnt = contours[0]
            # x, y, w, h = cv2.boundingRect(cnt)
            # result = result[y:y + h - 1, x:x + w - 1]
        # _, thresh = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
        # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnt = contours[0]
        # x, y, w, h = cv2.boundingRect(cnt)
        # result = result[y:y + h - 1, x:x + w - 1]
        # showAndWait(result)
        result = resize(result, 1000, 500)
        showAndWait(result)
        saveEnhancedImage(result,'result',algoName)
        print(algoName,totalTime)
        return totalTime,matchcount
    else:
        print('no enough images')



#--------------------------#


#
# stitch('SIFT', '٨', 40, True,'NONE')
# cv2.destroyAllWindows()
# stitch('SURF', '10', 40, True,'NONE')
# stitch('SIFT', '10', 40, True,'NONE')
stitch('ORB', '10', 100, True,'NONE')
stitch('SIFT', '10', 40, True,'NONE')
stitch('SURF', '10', 40, True,'NONE')
