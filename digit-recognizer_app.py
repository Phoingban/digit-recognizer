import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

def create_model(conv_layers, dense_layers, filters, kernel_size, pool_size):
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    
    current_size = 28
    for _ in range(conv_layers):
        if current_size < kernel_size:
            st.warning(f"Cannot add more layers. Current size ({current_size}) is smaller than kernel size ({kernel_size}).")
            break
        model.add(layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        if current_size // pool_size > 0:
            model.add(layers.MaxPooling2D(pool_size))
            current_size = current_size // pool_size
        else:
            st.warning(f"Skipping pooling layer. Current size ({current_size}) is too small for pooling size ({pool_size}).")
    
    model.add(layers.Flatten())
    
    for _ in range(dense_layers):
        model.add(layers.Dense(64, activation='relu'))
    
    model.add(layers.Dense(10, activation='softmax'))
    return model

def train_model(model, epochs, batch_size):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, verbose=0)
    return history

st.title('Digit Recognition CNN Parameter Tuning')

# Sidebar for parameters
# Sidebar for parameters
st.sidebar.header('Model Parameters')
conv_layers = st.sidebar.slider('Number of Convolutional Layers', 1, 3, 2, 
    help="Determines the depth of feature extraction. More layers can capture more complex patterns but may lead to overfitting.")
dense_layers = st.sidebar.slider('Number of Dense Layers', 1, 3, 1, 
    help="Affects the model's capacity to learn complex relationships. More layers increase capacity but may cause overfitting.")
filters = st.sidebar.slider('Number of Filters', 16, 64, 32, 16, 
    help="Controls the number of feature maps in each convolutional layer. More filters can detect more features but increase computational cost.")
kernel_size = st.sidebar.slider('Kernel Size', 2, 5, 3, 
    help="Defines the size of the convolutional filter. Larger sizes capture broader patterns but may lose fine details.")
pool_size = st.sidebar.slider('Pooling Size', 2, 4, 2, 
    help="Sets the size of the pooling operation, which reduces spatial dimensions. Larger sizes reduce more but may lose important information.")

st.sidebar.header('Training Parameters')
epochs = st.sidebar.slider('Number of Epochs', 1, 20, 5, 
    help="Specifies how many times the model will iterate over the entire dataset. More epochs allow more learning but may lead to overfitting.")
batch_size = st.sidebar.slider('Batch Size', 32, 256, 64, 32, 
    help="Determines how many samples are processed before the model is updated. Larger batches can lead to more stable updates but may converge to less optimal solutions.")

# Add a text input for the model name
model_name = st.sidebar.text_input('Model Name', 'mnist_model', 
    help="Enter a custom name for your model. It will be saved as '<your_name>.h5'")

if st.sidebar.button('Train Model'):
    with st.spinner('Training in progress...'):
        model = create_model(conv_layers, dense_layers, filters, kernel_size, pool_size)
        history = train_model(model, epochs, batch_size)
        
        # Plot training history
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='accuracy')
        ax.plot(history.history['val_accuracy'], label='val_accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        st.write(f'Test accuracy: {test_acc:.4f}')
        
        # Save model with custom name
        model_filename = f'{model_name}.h5'
        model.save(model_filename)
        st.success(f'Model trained and saved as {model_filename}')

        # Save the name of the most recently trained model
        with open('latest_model.txt', 'w') as f:
            f.write(model_filename)

# Add a new section for analyzing test dataset images
st.header('Analyze Test Dataset Images')

# Add a slider to select the number of random images to display
num_images = st.slider('Number of images to analyze', 1, 10, 5)

if st.button('Analyze Random Test Images'):
    # Check if a trained model exists
    if os.path.exists('latest_model.txt'):
        with open('latest_model.txt', 'r') as f:
            latest_model = f.read().strip()
        model = tf.keras.models.load_model(latest_model)
        st.write(f'Using model: {latest_model}')
    else:
        st.error('No trained model found. Please train a model first.')
        st.stop()

    # Select random images from the test dataset
    random_indices = random.sample(range(len(test_images)), num_images)
    
    for idx in random_indices:
        image = test_images[idx]
        true_label = test_labels[idx]
        
        # Reshape the image for the model
        image_for_model = tf.reshape(image, [1, 28, 28, 1])
        
        # Make prediction
        prediction = model.predict(image_for_model)
        predicted_digit = np.argmax(prediction)
        
        # Display the image and results
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        st.write(f'True label: {true_label}')
        st.write(f'Predicted digit: {predicted_digit}')
        st.write(f'Prediction confidence: {prediction[0][predicted_digit]:.2%}')
        st.write('---')

st.sidebar.header('Test Your Own Image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = plt.imread(uploaded_file)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, [28, 28])
    image = tf.reshape(image, [1, 28, 28, 1])
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Load the most recently trained model
    if os.path.exists('latest_model.txt'):
        with open('latest_model.txt', 'r') as f:
            latest_model = f.read().strip()
        model = tf.keras.models.load_model(latest_model)
        st.write(f'Using model: {latest_model}')
    else:
        st.error('No trained model found. Please train a model first.')
        st.stop()

    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    st.write(f'The predicted digit is: {predicted_digit}')