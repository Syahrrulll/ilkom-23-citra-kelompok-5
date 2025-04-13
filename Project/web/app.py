from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from konfigurasi import Konfigurasi

app = Flask(__name__)  # perbaiki _name ke __name__
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
            blue_value = None

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
            elif konversi == 'cek_biru':
                blue_channel = img[:, :, 0]
                blue_value = round(np.mean(blue_channel), 2)
                processed_img = img  

            # Jika tidak ada konversi yang menghasilkan gambar, gunakan gambar asli
            if processed_img is None:
                processed_img = img

            # Simpan hasil olahan
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            cv2.imwrite(processed_path, processed_img)

            # Buat histogram dari gambar hasil konversi
            histogram_filename = generate_histogram(processed_img, filename)

            return render_template('fitur.html', original=filename, processed=filename, histogram=histogram_filename, blue_value=blue_value)

    return render_template('fitur.html', original=None, processed=None, histogram=None, blue_value=None)


if __name__ == '__main__':  # perbaiki _name jadi __name__
    app.run(debug=True)
