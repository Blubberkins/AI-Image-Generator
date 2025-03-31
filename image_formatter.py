from PIL import Image
import os
import tensorflow as tf
from PIL import Image
import numpy as np

# Process images from an input folder into 256 x 256 pixel .png format and save them in the output folder
def preprocess_images(input_folder, output_folder, size=(256, 256)):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img_path = os.path.join(input_folder, filename)
            
            try:
                img = Image.open(img_path)

                # Handle transparency
                if img.mode == "P" or img.mode == "RGBA":
                    img = img.convert("RGBA")  # Convert palette/transparency images

                # Convert to RGB (removes transparency if needed)
                img = img.convert("RGB")
                
                # Resize to 256x256
                img = img.resize(size)

                # Save as PNG
                new_filename = os.path.splitext(filename)[0] + ".png"
                img.save(os.path.join(output_folder, new_filename), "PNG")

                print(f"✅ Processed: {filename} -> {new_filename}")

            except Exception as e:
                print(f"❌ Skipping {filename} due to error: {e}")

# Process a single image to fit the model input requirements
def preprocess_single_image(image_path, size=(256, 256)):
    # Open the image file
    img = Image.open(image_path).convert("RGB")
    
    # Resize to 256x256 pixels
    img = img.resize(size)
    
    # Convert to a numpy array
    img_array = np.array(img)
    
    # Normalize the image to [-1, 1] by scaling (since generator uses tanh activation)
    img_array = (img_array / 127.5) - 1.0
    
    # Convert to a Tensor and add batch dimension (shape: (1, 256, 256, 3))
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension
    
    return img_tensor

