# AI Image Generation File

# This file attempts to utilize the trained AI model to generate a new output image based off its learnings when given an input image in 'dataset/test'

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from image_formatter import preprocess_single_image
from image_generator import SelfAttention

# Load the trained generator model
generator = load_model('saved_model/generator.keras', custom_objects={'SelfAttention': SelfAttention})

# Function to save the generated image as a .png file
def save_generated_image(generated_image, output_path):

    # Rescale the image from [-1, 1] to [0, 255]
    generated_image = (generated_image + 1) * 127.5
    generated_image = tf.clip_by_value(generated_image, 0, 255)  # Ensure pixel values are in [0, 255]
    generated_image = generated_image.numpy().astype(np.uint8)  # Convert to uint8 for saving as image
    
    # Save the image as a .png file
    img = Image.fromarray(generated_image)
    img.save(output_path)
    print(f"Saved generated image at: {output_path}")

# Function to delete previous generated images
def delete_previous_outputs(test_dir):
    
    for file in os.listdir(test_dir):
        if file.startswith("dorofied_") and file.endswith(".png"):
            file_path = os.path.join(test_dir, file)
            os.remove(file_path)

# Function to process and generate output for the first image in dataset/test - change the 'test_dir' parameter if you wish to use different directory
def test_and_save_generated_image(test_dir='dataset/test'):

    # Delete old generated images
    delete_previous_outputs(test_dir)

    # Get the first image in the test directory
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    if not test_images:
        print("No images found in the test directory.")
        return
    
    first_image = test_images[0]
    input_image_path = os.path.join(test_dir, first_image)
    
    # Preprocess the image
    input_image = preprocess_single_image(input_image_path)
    
    # Generate the output image using the trained generator
    generated_image = generator(input_image, training=False)

    # Visualize the generated image
    plt.imshow(generated_image[0].numpy())
    plt.show()
    
    # Save the generated image as a .png file in the same directory
    output_image_path = os.path.join(test_dir, f"output_{first_image.split('.')[0]}.png")
    save_generated_image(generated_image[0], output_image_path)  # Save the first image in the batch

# Run the function to test and save the generated image
if __name__ == "__main__":
    test_and_save_generated_image()