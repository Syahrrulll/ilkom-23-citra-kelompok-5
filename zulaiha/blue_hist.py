import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png')
print(image.shape)
cv2.imshow('all',image)
cv2.imshow('blue',image[:,:,0])

only_blue = image[:,:,0]

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histogram Blue")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()