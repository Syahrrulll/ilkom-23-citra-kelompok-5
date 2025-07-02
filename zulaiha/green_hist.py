import cv2  # Library OpenCV untuk pengolahan gambar
import numpy as np  # Library NumPy untuk manipulasi array
import matplotlib.pyplot as plt  # Library Matplotlib untuk membuat grafik

# Membaca gambar berwarna dari file 'sample.png'
image = cv2.imread('sample.png')

# Mengambil channel warna hijau dari gambar
# Channel ke-1 mewakili warna hijau dalam format BGR OpenCV
only_green = image[:, :, 1]

# Menampilkan ukuran gambar (tinggi, lebar, jumlah channel warna)
print(image.shape)

# Menampilkan gambar asli (berwarna) dalam jendela berjudul 'all'
cv2.imshow('all', image)

# Menampilkan hanya channel hijau,(grayscale intensitas channel hijau)
cv2.imshow('green', only_green)

# [0, 256]: rentang nilai intensitas piksel
histogram = cv2.calcHist([only_green], [0], None, [256], [0, 256])

# Membuat figure baru untuk grafik histogram
plt.figure()
plt.title("Histogram Green")  # Judul grafik
plt.xlabel("Intensitas Piksel")  # Label sumbu X
plt.ylabel("Frekuensi")  # Label sumbu Y

# Menampilkan grafik histogram sebagai garis
plt.plot(histogram)

# Membatasi sumbu X dari 0 sampai 255
plt.xlim([0, 256])

# Menampilkan grafik
plt.show()
