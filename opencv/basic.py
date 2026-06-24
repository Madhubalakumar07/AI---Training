import cv2 as cv

#read img
img = cv.imread('Datasets/images/sample 3.jpg')
# cv.imshow('Image', img)

#gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

#blur
blur = cv.GaussianBlur(img, (15, 15), 0)
# cv.imshow('Blur', blur)

#edge
edge = cv.Canny(img, 150, 150)
# cv.imshow('Edge', edge)

#dilate
dil = cv.dilate(edge, None, iterations=1)
cv.imshow('Dilate', dil)

#erode
erode = cv.erode(dil, None, iterations=1)
cv.imshow('Erode', erode)

cv.waitKey(0)