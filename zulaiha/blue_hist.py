import cv2  # Library OpenCV untuk pengolahan citra
import numpy as np  # Library NumPy untuk manipulasi array
import matplotlib.pyplot as plt  # Library Matplotlib untuk plotting grafik

# Membaca gambar dari file 'sample.png'
image = cv2.imread('sample.png')

# Menampilkan ukuran (dimensi) gambar dalam format (tinggi, lebar, jumlah saluran warna)
print(image.shape)

# Menampilkan gambar asli dalam jendela berjudul 'all'
cv2.imshow('all', image)

# Menampilkan hanya saluran warna biru dari gambar (channel ke-0 dari format BGR OpenCV)
cv2.imshow('blue', image[:, :, 0])

# Menyimpan channel biru ke variabel only_blue
only_blue = image[:, :, 0]

# Menghitung histogram untuk channel biru (channel ke-0)
# Argumen [image]: input gambar
# Argumen [0]: channel biru
# None: tidak ada mask
# [256]: jumlah bin (0–255)
# [0, 256]: rentang nilai intensitas
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Membuat figure baru untuk histogram
plt.figure()
plt.title("Histogram Blue")  # Judul grafik
plt.xlabel("Intensitas Piksel")  # Label sumbu X
plt.ylabel("Frekuensi")  # Label sumbu Y

# Menampilkan grafik histogram
plt.plot(histogram)
plt.xlim([0, 256])  # Menentukan batas sumbu X dari 0 hingga 255

# Menampilkan grafik pada jendela terpisah
plt.show()
