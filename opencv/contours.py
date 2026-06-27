import cv2 as cv
import numpy as np

# read img
img = cv.imread('Datasets/images/sample 6.jpg')
cv.imshow('Image', img)
# gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

# canny
canny = cv.Canny(blur, 125, 170)
cv.imshow('Canny Image', canny)

blank = np.zeros(img.shape, dtype='uint8')
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)
cv.imshow('Contours Image', blank)

cv.waitKey(0)