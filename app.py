from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Convert image to pencil sketch
def pencil_sketch(image_path, contrast=1.0, thickness=1):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)

    # Adjust contrast
    sketch = np.clip(contrast * sketch, 0, 255).astype(np.uint8)

    # Adjust "pen thickness" (simulate stroke effect)
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        sketch = cv2.erode(sketch, kernel, iterations=1)

    return sketch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    contrast = float(request.form.get('contrast', 1.0))
    thickness = int(request.form.get('thickness', 1))

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    sketch = pencil_sketch(filepath, contrast, thickness)

    # Save the result temporarily
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sketch_' + file.filename)
    cv2.imwrite(result_path, sketch)

    return render_template('result.html', original=file.filename, result='sketch_' + file.filename)

if __name__ == "__main__":
    app.run(debug=True)
