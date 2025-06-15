import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
print(image.shape)
cv2.imshow('gray',image)


