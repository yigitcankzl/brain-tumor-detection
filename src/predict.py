import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

categories = ["glioma", "meningioma", "notumor", "pituitary"]

model = load_model('src/model/brain_tumor_detection_model.h5')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0  
    return img_array

def predict_image(img_path):
    img_array = prepare_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]
    return predicted_class

if __name__ == "__main__":
    img_path = input("Please enter the path to the image: ")
    
    if os.path.exists(img_path):
        prediction = predict_image(img_path)
        print(f"The model predicts the tumor type as: {prediction}")
    else:
        print("Invalid image path. Please check the file path and try again.")
