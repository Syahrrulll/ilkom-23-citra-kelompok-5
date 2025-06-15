import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png')
print(image.shape)
cv2.imshow('all',image)
cv2.imshow('blue',image[:,:,0])

only_blue = image[:,:,0]