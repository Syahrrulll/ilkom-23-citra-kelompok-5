from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rembg import remove  # Import remove dari rembg
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
                angle = float(request.form['rotate_angle'])
                center = (img.shape[1] // 2, img.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed_img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
            elif konversi == 'denoise':
                processed_img = cv2.medianBlur(img, 5)
            elif konversi == 'remove_bg':
                # Menghapus background dengan rembg
                with open(filepath, 'rb') as input_file:
                    input_data = input_file.read()
                    output_data = remove(input_data)
                
                # Simpan gambar hasil penghapusan background
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f"no_bg_{filename}")
                with open(processed_path, 'wb') as output_file:
                    output_file.write(output_data)
                processed_img = cv2.imread(processed_path)

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
