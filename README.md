# AI-Image-Generator

This program includes a functional AI image-to-image generation model based off Pix2Pix, taking in an image input and attempting to generate a corresponding image output based off the model's learnings from its trained dataset. The model can be trained to generate any desired output given the relevant training material, and uses various AI training techniques to improve on the model's training speed, accuracy and reliability. 

## How to use

Run the 'train_generator.py' to initialize the generator training program, which will format raw training data into a suitable format, train the AI image model on the given training data within the 'dataset' folder for 10 epochs, then save the model within the 'saved_model' folder. This file will need to be run every time the training dataset files change, and can be rerun as many times as needed if the user wishes to continually train the model to create better outputs. By default, training data should be placed into the 'dataset/raw' directory, where it will be processed into a valid, uniform format fit for training within the 'dataset/train' folder. This training data has no restrictions on file size and dimensions, so long as it matches one of the following standard image formats: .jpg, .png, or .webp. However, training data image pairs should match the following naming convention: <img1>_input, <img1>_output, <img2>_input, <img2>_output, etc.

Run the 'generate_image.py' file to utilise the trained model to generate an output picture. By default, the input image should be placed into the 'dataset/test' directory, where the output image produced by the generator will also appear with the format 'output_<input_image>.png'.

Note that the generator model currently included within this Github upload's 'saved_model' folder is trained off 97 different input/output image pairs, attempting to create 'Doro' caricature outputs from character image inputs, but is not able to generate its intended output due to a lack of training data.

## File structure

- dataset (Folder): Contains training data in the form of .jpg, .png and .webp image files, labelled in pairs of "imageX_input", "imageX_output", as well as the image outputs of the generator.
- saved_model (Folder): Contains the AI generator model
- dataset_loader.py: Contains functions which load the data from the dataset folder into a suitable dataset for the AI model to be trained on
- image_generator.py: Contains the main generator (Adversarial GAN) model and its training program, applying various techniques to improve its generation quality such as Self-Attention, regularization and learning rate optimizers. 
- generate_image.py: Used to operate the generator, accepting a single input image and utilising the trained generator model to attempt to produce a corresponding output image.
- image_formatter.py: Contains functions which format files such that they conform to a uniform format accepted by the generator model - specifically a 256x256 .png file.

## Detailed explanation of implementation


