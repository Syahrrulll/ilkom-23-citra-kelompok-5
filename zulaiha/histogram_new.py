import numpy as np  # Untuk operasi numerik dan array
import matplotlib.pyplot as plt  # Untuk menampilkan gambar dan grafik histogram
import os  # Untuk mengakses sistem file dan direktori
import cv2 as cv  # OpenCV untuk pemrosesan gambar

def histogramEqual():
    # Mendapatkan direktori kerja saat ini
    root = os.getcwd()

    # Menggabungkan path direktori kerja dengan nama file gambar
    imgPath = os.path.join(root, 'sample-histogram.png')

    # Membaca gambar dalam mode grayscale (hitam putih)
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    # Menghitung histogram dari gambar asli
    hist = cv.calcHist([img], [0], None, [256], [0,256])

    # Menghitung CDF (Cumulative Distribution Function) dari histogram
    cdf = hist.cumsum()

    # Menormalisasi CDF agar dapat ditampilkan di grafik bersama histogram
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    # Membuat figure (kanvas) untuk menampilkan hasil
    plt.figure()
    
    # Menampilkan gambar asli dalam format grayscale
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    # Menampilkan histogram dan CDF dari gambar asli
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.title("Histogram & CDF (Original)")
    plt.xlabel('Pixel Intensity')
    plt.ylabel('# of Pixels')