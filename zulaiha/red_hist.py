import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png')
print(image.shape)
cv2.imshow('all', image)
cv2.imshow('red', image[:,:,2])

histogram = cv2.calcHist([image], [2], None, [256], [0, 256])


