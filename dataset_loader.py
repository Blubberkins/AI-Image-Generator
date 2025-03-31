import tensorflow as tf
import os

# Dataset folder
TRAIN_DIR = "dataset/train"

# Function to augment input/output image pairs by applying random transformations
def augment_images(input_image, output_image):

    # Random horizontal flip (50% chance)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        output_image = tf.image.flip_left_right(output_image)

    # Random vertical flip (50% chance)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        output_image = tf.image.flip_up_down(output_image)

    # Random rotation (-10 to 10 degrees)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # Rotates by 90-degree multiples
    input_image = tf.image.rot90(input_image, k=k)
    output_image = tf.image.rot90(output_image, k=k)

    # Random brightness (mild)
    brightness_factor = tf.random.uniform([], -0.1, 0.1)
    input_image = tf.image.adjust_brightness(input_image, brightness_factor)
    output_image = tf.image.adjust_brightness(output_image, brightness_factor)

    # Random contrast
    contrast_factor = tf.random.uniform([], 0.7, 1.3)
    input_image = tf.image.adjust_contrast(input_image, contrast_factor)
    output_image = tf.image.adjust_contrast(output_image, contrast_factor)

    # Random saturation (only for RGB)
    if input_image.shape[-1] == 3:
        saturation_factor = tf.random.uniform([], 0.8, 1.2)
        input_image = tf.image.adjust_saturation(input_image, saturation_factor)
        output_image = tf.image.adjust_saturation(output_image, saturation_factor)

    # Random zoom & cropping (random zoom-in)
    scale = tf.random.uniform([], 0.8, 1.2)
    new_size = tf.cast(tf.cast(tf.shape(input_image)[:2], tf.float32) * scale, tf.int32)
    input_image = tf.image.resize(input_image, new_size)
    output_image = tf.image.resize(output_image, new_size)
    input_image = tf.image.resize_with_crop_or_pad(input_image, 256, 256)
    output_image = tf.image.resize_with_crop_or_pad(output_image, 256, 256)

    # Add small elastic distortion (Random Affine Transformations)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.central_crop(input_image, central_fraction=tf.random.uniform([], 0.9, 1.0))
        output_image = tf.image.central_crop(output_image, central_fraction=tf.random.uniform([], 0.9, 1.0))

    return input_image, output_image

# Function to load and process paired images
def load_image_pair(filename):

    input_path = tf.strings.join([TRAIN_DIR, '/', filename, '_input.png'])
    output_path = tf.strings.join([TRAIN_DIR, '/', filename, '_output.png'])

    # Read and process the input and output images
    input_image = tf.io.read_file(input_path)
    input_image = tf.image.decode_png(input_image, channels=3)
    output_image = tf.io.read_file(output_path)
    output_image = tf.image.decode_png(output_image, channels=3)

    # Apply augmentations
    input_image, output_image = augment_images(input_image, output_image)

    # Resize input and output images
    input_image = tf.image.resize(input_image, [256, 256])
    output_image = tf.image.resize(output_image, [256, 256])

    # Normalize input and output images to [-1, 1]
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    output_image = tf.cast(output_image, tf.float32) / 127.5 - 1

    return input_image, output_image

# Create a dataset of paired images for training using a batch size of 8 (ADJUSTABLE)
def create_dataset(batch_size=8):
    file_names = []
    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith("_input.png"):
            base_name = filename.replace("_input.png", "")
            file_names.append(base_name)

    file_names = sorted(file_names)  # Ensure consistent ordering
    dataset = tf.data.Dataset.from_tensor_slices(file_names)  # Convert list to tensor slices
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)  # Map image pairs
    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Optimize data loading

    return dataset