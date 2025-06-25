from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove
from konfigurasi import Konfigurasi
from collections import Counter
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['HISTOGRAM_FOLDER'] = 'static/histogram/'

# Buat folder jika belum ada
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'], app.config['HISTOGRAM_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

def generate_histogram(image, filename):
    histogram_filename = f"hist_{filename}.png"
    histogram_path = os.path.join(app.config['HISTOGRAM_FOLDER'], histogram_filename)
    plt.figure(figsize=(6, 4))
    if len(image.shape) == 2:
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title('Grayscale Histogram')
    else:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            plt.plot(cv2.calcHist([image], [i], None, [256], [0, 256]), color=col)
        plt.title('Color Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()
    return histogram_filename

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    counts = Counter([tuple(pixel) for pixel in pixels])
    dominant_color = counts.most_common(1)[0][0]
    return dominant_color

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/config_spk', methods=['GET', 'POST'])
def config_spk():
    konfig = Konfigurasi()
    data = konfig.read_yaml()
    if request.method == "POST":
        key = request.form["key"]
        value = request.form["value"]
        konfig.update_yaml(key, value)
        return redirect(url_for("config_spk"))
    return render_template("konfigurasi.html", data=data)

@app.route('/history')
def history():
    uploads = os.listdir(app.config['UPLOAD_FOLDER'])
    processed = os.listdir(app.config['PROCESSED_FOLDER'])
    return render_template('history.html', uploads=uploads, processed=processed)

@app.route('/proses_gambar', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        konversi = request.form.get('konversi')
        angle = float(request.form.get('rotate_angle', 90))

        if not file or file.filename == '':
            return redirect(request.url)

        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        processed_img = None
        histogram_filename = None
        channel_name = None
        color_value = None

        if konversi == 'grayscale':
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif konversi == 'edge':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.Canny(gray, 100, 200)
        elif konversi == 'blur':
            processed_img = cv2.GaussianBlur(img, (11, 11), 0)
        elif konversi == 'invert':
            processed_img = cv2.bitwise_not(img)
        elif konversi == 'threshold':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, processed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif konversi == 'bit1':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, processed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif konversi == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_img = cv2.filter2D(img, -1, kernel)
        elif konversi == 'face_detect':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            processed_img = img
        elif konversi == 'rotate':
            center = (img.shape[1] // 2, img.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            processed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        elif konversi == 'denoise':
            processed_img = cv2.medianBlur(img, 5)
        elif konversi == 'remove_bg':
            with open(filepath, 'rb') as input_file:
                output_data = remove(input_file.read())
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"no_bg_{filename}")
            with open(processed_path, 'wb') as output_file:
                output_file.write(output_data)
            processed_img = cv2.imread(processed_path)
        elif konversi == 'dominant_color':
            color = get_dominant_color(img)
            color_value = f"RGB{color}"
            channel_name = "Dominant Color"
            processed_img = np.zeros_like(img)
            processed_img[:, :] = color
        elif konversi == 'pencil_sketch':
            gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            processed_img = sketch
        elif konversi == 'sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(img, sepia_filter)
            processed_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

        elif konversi == 'cartoon':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img, 9, 300, 300)
            processed_img = cv2.bitwise_and(color, color, mask=edges)

        elif konversi == 'brightness':
            value = 50  # Tambah brightness +50
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = np.clip(v + value, 0, 255)
            final_hsv = cv2.merge((h, s, v))
            processed_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        elif konversi == 'contrast':
            alpha = 1.5  # Tingkat kontras
            beta = 0     # Tidak mengubah brightness
            processed_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        elif konversi == 'emboss':
            kernel = np.array([[ -2, -1, 0],
                               [ -1, 1, 1],
                               [  0, 1, 2]])
            processed_img = cv2.filter2D(img, -1, kernel)


        if processed_img is None:
            processed_img = img

        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cv2.imwrite(processed_path, processed_img)
        histogram_filename = generate_histogram(processed_img, filename)

       return render_template('fitur.html',
                               original=filename,
                               processed=filename,
                               histogram=histogram_filename,
                               blue_value=color_value,
                               channel_name=channel_name)

    return render_template('fitur.html', original=None, processed=None, histogram=None, blue_value=None)

if __name__ == '__main__':
    app.run(debug=True)
