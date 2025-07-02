import cv2  # Library OpenCV untuk pemrosesan gambar
import numpy as np  # Library NumPy untuk manipulasi data numerik dan array
import matplotlib.pyplot as plt  # Library Matplotlib untuk membuat grafik

# Membaca gambar dalam mode grayscale (hitam-putih)
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

# Menampilkan dimensi gambar dalam format (tinggi, lebar)
print(image.shape)

# Menampilkan gambar grayscale dengan jendela berjudul 'gray'
cv2.imshow('gray', image)

# [0, 256]: rentang nilai intensitas piksel
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Membuat figure untuk menampilkan grafik histogram
plt.figure()
plt.title("Histogram Grayscale")  # Judul grafik
plt.xlabel("Intensitas Piksel")  # Label sumbu X (nilai 0-255)
plt.ylabel("Frekuensi")  # Label sumbu Y (jumlah piksel untuk tiap intensitas)

# Menampilkan histogram sebagai grafik garis
plt.plot(histogram)
plt.xlim([0, 256])  # Menentukan batas sumbu X dari 0 hingga 255

# Menampilkan grafik histogram
plt.show()
