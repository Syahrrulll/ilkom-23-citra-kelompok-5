import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png')
print(image.shape)
cv2.imshow('all', image)
cv2.imshow('red', image[:,:,2])

histogram = cv2.calcHist([image], [2], None, [256], [0, 256])

plt.figure()
plt.title("Histogram Red")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.plot(histogram, color='red')
plt.xlim([0, 256])
plt.show()
