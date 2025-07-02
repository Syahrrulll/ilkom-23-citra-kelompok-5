import numpy as np  # Untuk operasi numerik dan array
import matplotlib.pyplot as plt  # Untuk menampilkan gambar dan grafik histogram
import os  # Untuk mengakses sistem file dan direktori
import cv2 as cv  # OpenCV untuk pemrosesan gambar

def histogramEqual():
    # Untuk mendapatkan direktori kerja saat ini
    root = os.getcwd()

    # Untuk Menggabungkan path direktori kerja dengan nama file gambar
    imgPath = os.path.join(root, 'sample-histogram.png')

    # Untuk Membaca gambar dalam mode grayscale (hitam putih)
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # Untuk Menghitung histogram dari gambar asli
    hist = cv.calcHist([img], [0], None, [256], [0,256])

    # Untuk Menghitung CDF (Cumulative Distribution Function) dari histogram
    cdf = hist.cumsum()

    # Untuk Menormalisasi CDF agar dapat ditampilkan di grafik bersama histogram
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    # Untuk Membuat figure (kanvas) untuk menampilkan hasil
    plt.figure()
    
    # Untuk Menampilkan gambar asli dalam format grayscale
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Untuk Menampilkan histogram dan CDF dari gambar asli
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.title("Histogram & CDF (Original)")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')
    
    # Untuk Melakukan histogram equalization pada gambar asli
    equImg = cv.equalizeHist(img)

    # Untuk Menghitung histogram dari gambar yang telah di-equalize
    equhist = cv.calcHist([equImg], [0], None, [256], [0,256])

    # Untuk Menghitung dan menormalkan CDF dari gambar equalized
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()

    # UntukMenampilkan gambar hasil histogram equalization
    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.title("Equalized Image")

    # Untuk Menampilkan histogram dan CDF dari gambar equalized
    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(equcdfNorm, color='b')
    plt.title("Histogram & CDF (Equalized)")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')

    # Untuk Membuat objek CLAHE (Contrast Limited Adaptive Histogram Equalization)
    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))

    # Untuk Menerapkan CLAHE pada gambar asli
    claheImg = claheObj.apply(img)

    # Untuk Menghitung histogram dari hasil CLAHE
    clahehist = cv.calcHist([claheImg], [0], None, [256], [0,256])

    # Untuk Menghitung dan menormalkan CDF dari hasil CLAHE
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()
    
    # Untuk Menampilkan gambar hasil CLAHE
    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.title("CLAHE Image")
