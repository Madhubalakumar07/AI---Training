import cv2 as cv
import numpy as np

#read img
img = cv.imread('Datasets/images/sample 6.jpg')
# cv.imshow('Image', img)

# translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dim = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dim)

trans = translate(img, 150, 100)
# cv.imshow('Translated Image', trans)

# rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    return cv.warpAffine(img, rotMat, (width, height))

rot = rotate(img, 90, (400,400))
cv.imshow('Rotated Image', rot)

# flipping
flip = cv.flip(img, 1)
# cv.imshow('Flipped Image', flip)

cv.waitKey(0)