from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from konfigurasi import Konfigurasi

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'
app.config['HISTOGRAM_FOLDER'] = 'static/histogram/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTOGRAM_FOLDER'], exist_ok=True)


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


@app.route('/proses_gambar', methods=['GET', 'POST'])
def upload_image():
    def generate_histogram(image, filename):
        histogram_filename = f"hist_{filename}.png"
        histogram_path = os.path.join(app.config['HISTOGRAM_FOLDER'], histogram_filename)
        plt.figure(figsize=(6, 4))

        if len(image.shape) == 2:  # grayscale
            plt.hist(image.ravel(), bins=256, range=[0, 256], color='gray')
            plt.title('Grayscale Histogram')
        else:  # RGB
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

    if request.method == 'POST':
        if 'image' not in request.files or 'konversi' not in request.form:
            return redirect(request.url)

        file = request.files['image']
        konversi = request.form['konversi']

        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)


            img = cv2.imread(filepath)
            processed_img = None
            histogram_filename = None
            color_value = None
            channel_name = None

            # Pilihan konversi
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
            elif konversi in ['cek_biru', 'cek_merah', 'cek_hijau']:
                if konversi == 'cek_biru':
                    target = np.array([255, 0, 0])  # BGR
                    channel_index = 0
                    channel_name = "Biru"
                elif konversi == 'cek_hijau':
                    target = np.array([0, 255, 0])
                    channel_index = 1
                    channel_name = "Hijau"
                elif konversi == 'cek_merah':
                    target = np.array([0, 0, 255])
                    channel_index = 2
                    channel_name = "Merah"

                distance = np.linalg.norm(img.astype(np.int16) - target, axis=2)
                max_distance = np.sqrt((255**2) * 3)
                intensity = 255 - (distance / max_distance * 255).astype(np.uint8)
                processed_img = intensity

                channel = img[:, :, channel_index]
                color_value = round(np.mean(channel), 2)

            if processed_img is None:
                processed_img = img

            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            cv2.imwrite(processed_path, processed_img)

            histogram_filename = generate_histogram(processed_img, filename)

            return render_template(
                'fitur.html',
                original=filename,
                processed=filename,
                histogram=histogram_filename,
                blue_value=color_value,
                channel_name=channel_name,
                color_value=color_value
            )

    return render_template('fitur.html', original=None, processed=None, histogram=None, blue_value=None)


if __name__ == '__main__':
    app.run(debug=True)
