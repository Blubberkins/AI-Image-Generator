# AI Image Generator Training File

# This file initializes an AI generator model (based on Pix2Pix) and trains it using pairs of images (labelled imgX_input/imgX_output) within the 'dataset/train' folder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from keras.saving import register_keras_serializable
import os
from image_formatter import preprocess_images
from dataset_loader import create_dataset

# Paths to dataset
INPUT_DIR = "dataset/raw"  # Folder containing all raw images
OUTPUT_DIR = "dataset/train"  # Folder where processed images will be saved

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a Self-Attention block, allowing the model to capture long-range dependencies, improving detail and consistency in the output
@register_keras_serializable()
class SelfAttention(layers.Layer):

    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels

        # Define the layers
        self.query = layers.Conv2D(channels // 8, kernel_size=1)
        self.key = layers.Conv2D(channels // 8, kernel_size=1)
        self.value = layers.Conv2D(channels, kernel_size=1)
        self.gamma = self.add_weight(shape=[1], initializer="zeros", trainable=True)

    def build(self, input_shape):
        # Ensure that the layer is built properly by explicitly specifying the shape
        self.query.build(input_shape)
        self.key.build(input_shape)
        self.value.build(input_shape)
        super(SelfAttention, self).build(input_shape)  # Call the parent build method

    @tf.function
    def call(self, x):
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channels = tf.shape(x)[3]

        # Compute the query, key, and value
        q = tf.reshape(self.query(x), (batch_size, height * width, self.channels // 8))
        k = tf.reshape(self.key(x), (batch_size, self.channels // 8, height * width))
        v = tf.reshape(self.value(x), (batch_size, height * width, channels))

        # Compute attention
        attention = tf.nn.softmax(tf.matmul(q, k), axis=-1)
        attention_output = tf.matmul(attention, v)
        attention_output = tf.reshape(attention_output, (batch_size, height, width, channels))

        return self.gamma * attention_output + x
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            "channels": self.channels,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        # Define the output shape (same as input shape for SelfAttention)
        return input_shape

# Define L2 regularization value (ADJUSTABLE)
regularizer = l2(1e-4)

# Downsampling block
def downsample(filters, size, apply_batchnorm=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    conv_layer = layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False,
                                kernel_regularizer=regularizer) # Use L2 regularization
    
    # Apply spectral normalization to the Conv2D layer
    result.add(tf.keras.layers.SpectralNormalization(conv_layer))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result

# Upsampling block
def upsample(filters, size, apply_dropout=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5)) # Uses dropout regularization (ADJUSTABLE)

    result.add(layers.ReLU())

    return result

# Generator model
def build_generator():

    inputs = layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        SelfAttention(512),  # Add self-attention after deep downsampling
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')

    x = inputs

    # Downsampling
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling
    for up, skip in zip(up_stack, skips):
        x = up(x)

        # Get static shape of 'x' to ensure compatibility
        x_shape = tf.keras.backend.int_shape(x)
        # Resize skip connection to match 'x'
        skip_resized = layers.Resizing(x_shape[1], x_shape[2])(skip)
        # Concatenate the upsampled skip tensor with the output of the upsampling block
        x = layers.Concatenate()([x, skip_resized])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

# Discriminator model
def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[256, 256, 3], name="input_image")
    tar = layers.Input(shape=[256, 256, 3], name="target_image")

    x = layers.Concatenate()([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    sa = SelfAttention(256)(down3)  # Add self-attention before final classification

    zero_pad1 = layers.ZeroPadding2D()(sa)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer, use_bias=False)(zero_pad1)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)

    return keras.Model(inputs=[inp, tar], outputs=[last, sa])

# Define loss functions
loss_object = keras.losses.BinaryCrossentropy(from_logits=True)

# Loss function for the generator that combines adversarial and L1 loss, with L1 = 50 (ADJUSTABLE)
def generator_loss(disc_generated_output, gen_output, target, lambda_l1=50):

    # Adversarial loss (GAN loss) - Encourage realistic images
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # L1 loss (Pixel-wise absolute difference) - Ensure structural similarity
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Weighted sum of GAN loss and L1 loss
    total_loss = gan_loss + (lambda_l1 * l1_loss)
    
    return total_loss

# Loss function for the discriminator that uses label smoothing and feature matching loss
def discriminator_loss(disc_real_output, disc_generated_output, real_features, fake_features):

    real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output) * 0.9, disc_real_output) # Label smoothing applied (0.9 instead of 1.0)
    
    fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output)
    
    feature_matching_loss = tf.reduce_mean(tf.abs(real_features - fake_features))  # L1 loss on feature maps
    
    total_disc_loss = real_loss + fake_loss + 0.1 * feature_matching_loss  # Scale feature matching loss
    return total_disc_loss


# Optimizers - uses learning rate scheduler to reduce learning rate over time, improving stability
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4, decay_steps=10000, decay_rate=0.95, staircase=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

# Training Function - determines how the model is trained
@tf.function
def train_step(input_image, target, generator, discriminator):

    # Add small Gaussian noise (prevents overfitting)
    noise = tf.random.normal(shape=tf.shape(input_image), mean=0.0, stddev=0.05)
    input_image = input_image + noise

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        # Get discriminator outputs
        disc_real_output, real_features = discriminator([input_image, target], training=True)
        disc_generated_output, fake_features = discriminator([input_image, gen_output], training=True)

        # Compute loss
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, real_features, fake_features)

    # Compute gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Main executable - runs image preprocessing, loads the dataset
def train():

    # Run preprocessing (COMMENTED OUT UNTIL DATASET CHANGES)
    #preprocess_images(INPUT_DIR, OUTPUT_DIR)

    # Load dataset using batches of 8 (ADJUSTABLE)
    train_dataset = create_dataset(batch_size=8)  # Use dataset loader file

    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile generator
    generator.compile(optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0.5),
                    loss=tf.keras.losses.MeanAbsoluteError())

    # Training loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        for input_image, target in train_dataset:
            gen_loss, disc_loss = train_step(input_image, target, generator, discriminator)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")

    # Save the trained model
    MODEL_SAVE_PATH = "saved_model/generator.keras"
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    generator.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()