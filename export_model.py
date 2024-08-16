import tensorflow as tf
import os

# Print current directory and files
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())

# Path to your model
model_path = 'model.h5'  # Update this with your actual model name

try:
    # Load your trained model
    print(f"Attempting to load model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    # Convert the model
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Model converted successfully")

    # Save the model
    tflite_path = 'mnist_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")

except Exception as e:
    print(f"An error occurred: {str(e)}")