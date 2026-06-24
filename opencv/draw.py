import cv2 as cv
import numpy as np


img = np.zeros((512, 512, 3), np.uint8)
cv.imshow('Black', img)

# paint in colour
img[100:200,0:100] = [255, 0, 0]
cv.imshow('Blue', img)

#rectangle
cv.rectangle(img, (100,200), (500,500), (0,255,0), 2)
cv.imshow('Rectangle', img)

#circle
cv.circle(img, (256, 256), 50, (0,0,255), -1)
cv.imshow('Circle', img)

#line
cv.line(img, (0,0), (512,512), (255,255,255), 5)
cv.imshow('Line', img)

#text
cv.putText(img, 'evening cunts', (100, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv.imshow('Text', img)

cv.waitKey(0)