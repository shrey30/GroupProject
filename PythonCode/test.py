import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Blood1.keras')
image_path = os.path.join(script_dir, '444.jpg')

# Verify paths exist
print(f"Model path: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Image path: {image_path}")
print(f"Image file exists: {os.path.exists(image_path)}")

# Load the saved model
model = tf.keras.models.load_model(model_path)
x=450
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(x, x))  # Adjust size to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class, predictions

className=['Eosinophil','Lymphocyte','Basophils','Erythroblast']
#className=['Basophils','Eosinophil','Erythroblast','Lymphocyte']
# Make a prediction
predicted_class, predictions = predict_image(image_path)
print(predicted_class)
# Display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {className[predicted_class[0]]}')
plt.show()

# Show the prediction probabilities for each class
print('Prediction probabilities:', predictions)
