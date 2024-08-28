# Digit Recognizer

## Overview
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. It features a Streamlit web application for interactive model training, parameter tuning, and digit recognition.

## Features
- Interactive CNN model training with customizable parameters
- Real-time visualization of training progress
- Evaluation on test dataset
- Analysis of random test images
- Custom image upload for digit recognition
- Model export to TensorFlow Lite format


## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/digit-recognizer.git
   digit-recognizer
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run digit-recognizer_app.py
   ```

2. Use the sidebar to adjust model parameters and train the CNN.
3. Analyze random test images or upload your own image for digit recognition.

## Model Architecture
The CNN architecture is dynamically created based on user-defined parameters:
- Convolutional layers: 1-3
- Dense layers: 1-3
- Filters: 16-64
- Kernel size: 2-5
- Pooling size: 2-4

## Model Export
To export the trained model to TensorFlow Lite format:
1. Ensure you have trained a model and it's saved as an .h5 file.
2. Update the `model_path` in `export_model.py` to match your saved model's filename.
3. Run the export script:
   ```
   python export_model.py
   ```



