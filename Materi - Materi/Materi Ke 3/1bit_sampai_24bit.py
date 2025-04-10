import numpy as np
import matplotlib.pyplot as plt

# membuat array image_1: representasi gambar grayscale (5x7 piksel)
image_1 = np.array([
    [0,0,255,0,35,0,0],
    [0,210,0,25,0,50,0],
    [0,200,0,0,0,80,0],
    [0,0,180,0,120,0,0],
    [0,0,0,140,0,0,0],
], dtype=np.uint8)

# Membuat array image_2 (tidak ditampilkan di plot saat ini)
image_2 = np.array([
    [0,255,200],
    [130,3,210],
    [100,4,180]  
], dtype=np.uint8)

# Membuat array creeper: representasi citra RGB 8x12 (mirip sprite karakter)
creeper = np.array([
    [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[255,0,0],[255,0,0],[0,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
], dtype=np.uint8)

# Menyiapkan subplot 1 baris 2 kolom untuk menampilkan dua gambar
_, axs = plt.subplots(1,2, figsize=(10, 5))

# Menampilkan gambar grayscale di sebelah kiri
axs[0].imshow(image_1, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("uji coba citra")
# axs[1].imshow(image_2, cmap='gray', vmin=0, vmax=255)
# axs[1].set_title("uji coba citra")

# Menampilkan gambar RGB 'creeper' di sebelah kanan
axs[1].imshow(creeper)
axs[1].set_title("creeper??")

plt.show()


