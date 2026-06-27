import cv2 as cv
import matplotlib.pyplot as plt

# read img
img = cv.imread('Datasets/images/sample 6.jpg')

# gray scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Image', gray)

# hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV Image', hsv)

# lab
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB Image', lab)

# bgr to rgb
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB Image', rgb)
plt.imshow(rgb)
plt.show()

cv.waitKey(0)