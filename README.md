MNIST Digit Recognizer
Overview
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. It features a Streamlit web application for interactive model training, parameter tuning, and digit recognition.
Features

Interactive CNN model training with customizable parameters
Real-time visualization of training progress
Evaluation on test dataset
Analysis of random test images
Custom image upload for digit recognition
Model export to TensorFlow Lite format

Requirements

Python 3.7+
TensorFlow 2.x
Streamlit
NumPy
Matplotlib

Installation

Clone this repository:
Copygit clone https://github.com/yourusername/mnist-digit-recognizer.git
cd mnist-digit-recognizer

Install the required packages:
Copypip install -r requirements.txt


Usage

Run the Streamlit app:
Copystreamlit run digit-recognizer_app.py

Use the sidebar to adjust model parameters and train the CNN.
Analyze random test images or upload your own image for digit recognition.

Model Architecture
The CNN architecture is dynamically created based on user-defined parameters:

Convolutional layers: 1-3
Dense layers: 1-3
Filters: 16-64
Kernel size: 2-5
Pooling size: 2-4

Model Export
To export the trained model to TensorFlow Lite format:

Ensure you have trained a model and it's saved as an .h5 file.
Update the model_path in export_model.py to match your saved model's filename.
Run the export script:
Copypython export_model.py


Contributing
Contributions to improve the project are welcome. Please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
Distributed under the MIT License. See LICENSE for more information.
Contact
Your Name - @yourtwitter - email@example.com
Project Link: https://github.com/yourusername/mnist-digit-recognizer
