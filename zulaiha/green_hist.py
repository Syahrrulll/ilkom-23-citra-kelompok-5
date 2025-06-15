import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('sample.png')
only_green = image[:,:,1]

print(image.shape)

cv2.imshow('all',image)
cv2.imshow('green',only_green)

histogram = cv2.calcHist([only_green], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histogram green")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()
# cv2.waitK
