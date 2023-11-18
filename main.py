from fastapi import FastAPI, UploadFile
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import tensorflow as tf
import io  # Add this line to import the io module

app = FastAPI()

# Lazy loading of the model during startup
model = None

def load_keras_model():
    global model
    if model is None:
        model = load_model('model_vgg19.h5')

# Predict function with image resizing and direct processing
def predict(img_data):
    load_keras_model()
    
    img = image.img_to_array(img_data)
    img = np.expand_dims(img, axis=0)
    img_data = preprocess_input(img)
    classes = model.predict(img_data)
    malignant = float(classes[0, 0])  # Convert to float
    normal = float(classes[0, 1])     # Convert to float
    
    return malignant, normal

@app.post("/predict/")
async def predict_image(file: UploadFile):
    # Read image file directly without saving
    contents = await file.read()
    img_data = image.load_img(io.BytesIO(contents), target_size=(224, 224))
    
    # Perform prediction on the image data
    malignant, normal = predict(img_data)
    
    # Convert NumPy floats to Python floats
    malignant = float(malignant)
    normal = float(normal)
    
    if malignant > normal:
        prediction = 'malignant'
    else:
        prediction = 'normal'
    
    return {"prediction": prediction, "malignant_prob": malignant, "normal_prob": normal}
