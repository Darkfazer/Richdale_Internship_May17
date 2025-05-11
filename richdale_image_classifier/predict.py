import tensorflow as tf 
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Load pre-trained MobileNet model
model = MobileNet(weights='imagenet')

def classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    
    print("Predictions:")
    for i, (imagenetID, label, prob) in enumerate(decoded):
        print(f"{i+1}. {label}: {prob*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image.jpg")
    else:
        classify(sys.argv[1])
