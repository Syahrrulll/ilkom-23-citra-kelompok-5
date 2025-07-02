import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv

def histogramEqual():
    # Mengambil direktori kerja saat ini
    root = os.getcwd()

    # Membuat path gambar yang akan diproses
    imgPath = os.path.join(root, 'sample-histogram.png')

    # Membuat path gambar yang akan diproses
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # Menghitung histogram dari gambar asli
    hist = cv.calcHist([img], [0], None, [256], [0,256])

    # Menghitung cumulative distribution function (CDF) dari histogram
    cdf = hist.cumsum()

    # Menormalisasi CDF agar bisa digambarkan di grafik
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    # Membuat figure untuk menampilkan gambar dan grafik
    plt.figure()

    # Menampilkan gambar asli
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Menampilkan histogram dan CDF dari gambar asli
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.title('Original Histogram & CDF')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')

    # Melakukan histogram equalization (penyeimbangan histogram) pada gambar asli
    equImg = cv.equalizeHist(img)

    # Menghitung histogram dari gambar yang telah di-equalize
    equhist = cv.calcHist([equImg], [0], None, [256], [0,256])

    # Menghitung CDF dari histogram hasil equalization
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()

    # Menampilkan gambar setelah histogram equalization
    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.title('Equalized Image')

    # Menampilkan histogram dan CDF dari gambar hasil equalization
    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(equcdfNorm, color='b')
    plt.title('Equalized Histogram & CDF')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')

    # Membuat objek CLAHE (Contrast Limited Adaptive Histogram Equalization)
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))

    # Menerapkan CLAHE pada gambar asli
    claheImg = claheObj.apply(img)

    # Menghitung histogram dari gambar hasil CLAHE
    clahehist = cv.calcHist([claheImg], [0], None, [256], [0,256])

    # Menghitung CDF dari histogram hasil CLAHE
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()

    # Menampilkan gambar hasil CLAHE
    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.title('CLAHE Image')

    # Menampilkan histogram dan CDF dari gambar hasil CLAHE
    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm, color='b')
    plt.title('CLAHE Histogram & CDF')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')

    # Menampilkan semua gambar dan grafik
    plt.tight_layout()
    plt.show()

# Memanggil fungsi untuk menjalankan proses dan menampilkan hasil
histogramEqual()
