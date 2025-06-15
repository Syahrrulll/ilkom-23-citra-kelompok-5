import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

def histogramEqual():
    root = os.getcwd()
    imgPath = os.path.join(root, 'sample-histogram.png')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.xlabel('pixel intesity')
    plt.ylabel('# of pixels')
    equImg = cv.equalizeHist(img)
    equhist = cv.calcHist([equImg], [0], None, [256], [0,256])
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()