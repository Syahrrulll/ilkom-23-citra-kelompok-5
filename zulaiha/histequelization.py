import cv2
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)