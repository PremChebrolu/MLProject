import base64
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
   return render_template('index.html')

@app.route("/api/upload", methods=['POST'])
def upload():
    model = tf.keras.models.load_model('model.keras')
    if request.method == 'POST':
        #print(request.json)
        img = request.json['image'].replace('data:image/png;base64,', '')
        imgdata = base64.b64decode(img)
        filename = 'uploads/image.png'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        size = (28, 28)
        image = Image.open(filename)
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0,0), image)
        new_image.convert('RGBA')
        new_image = new_image.resize(size)
        new_image.save('uploads/image.png')
        
        inp = cv2.imread('uploads/image.png')[:,:,0]
        #print(inp)
        #print(len(inp), len(inp[0]))
        inp = np.invert(np.array([inp]))
        inp = tf.keras.utils.normalize(inp)
        #print(inp)
        prediction = model.predict(inp)
        #print(f"This digit is probably a {np.argmax(prediction)}")
        data = {'prediction': f"{np.argmax(prediction)}"}
    return data, 200
if __name__ == '__main__':
   app.run()