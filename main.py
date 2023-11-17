from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, "*" can be replaced with your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your Keras model
model = load_model('model_vgg19.h5')

def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    malignant = float(classes[0, 0])  # Convert to float
    normal = float(classes[0, 1])     # Convert to float
    
    return malignant, normal

@app.post("/predict/")
async def predict_image(file: UploadFile):
    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as temp_image:
        temp_image.write(file.file.read())
    
    # Perform prediction on the saved image
    malignant, normal = predict("temp_image.jpg")
    
    # Clean up the temporary image file
    import os
    os.remove("temp_image.jpg")
    
    if malignant > normal:
        prediction = 'malignant'
    else:
        prediction = 'normal'
    
    # Convert NumPy floats to Python floats
    malignant = float(malignant)
    normal = float(normal)
    
    return {"prediction": prediction, "malignant_prob": malignant, "normal_prob": normal}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )
