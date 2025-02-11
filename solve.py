from flask import jsonify
import base64
import numpy as np
import cv2
from keras.models import load_model

# Load the model
# tb_model = load_model("tb_classifier.h5")

def pred(data):
    try:
        # Decode Base64 image
        image_data = base64.b64decode(data["image"])
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Load image in RGB

        if image is None:
            return {"error": "Failed to process image"}

        # Resize to match model input (224x224)
        image = cv2.resize(image, (224, 224))

        # Convert BGR to RGB (since OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values (if needed, depends on model training)
        image = image / 255.0  # Normalize to range [0, 1]

        # Reshape to (1, 224, 224, 3) for model
        image = np.expand_dims(image, axis=0)

        # Model prediction
        # prediction = tb_model.predict(image)
        prediction = prediction.tolist()  # Convert NumPy array to JSON serializable list

        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

# Read an image and convert it to Base64
# with open("6.png", "rb") as image_file:  
#     base64_string = base64.b64encode(image_file.read()).decode("utf-8")

# # Test the function
# result = pred({"image": base64_string})
# print("Model prediction:", result)
# import os
# print(os.listdir() )
      
# symptoms_pred = [1,0,1,0,1,0,1,0,1,0]
# pred_nn = load_model("symp_nn.h5")
# pred_rf = load_model("symp_rf.h5")
# pred_1 = pred_nn.predict([symptoms_pred])
# pred_2 = pred_rf.predict([symptoms_pred])
# pred_3 = (pred_1 + pred_2) / 2
# print(pred_1,"********",pred_2,"********",pred_3,"********")

