from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model("MyModel.h5")

@app.route('/')
def index():
    return "Ear Based Gender Classifcation"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        np_image = Image.open(file)
        np_image = np.array(np_image).astype('float32')/255
        np_image = transform.resize(np_image, (256,256, 3))
        np_image = np.expand_dims(np_image, axis=0)
        preds = model.predict(np_image)
        if preds[0][0] > 0.5:
            result = "Female"
        else:
            result = "Male"
        return result

if __name__ == '__main__':
    app.run(host ="0.0.0.0",debug=True)
