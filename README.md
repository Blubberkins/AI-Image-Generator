# AI-Doro-Generator

This is a program which includes a functional AI image-to-image generation model based off Pix2Pix, taking in an image input and attempting to generate a corresponding image output based off the model's learnings from its trained dataset. Originally developed to generate caricatures based off the original 'Doro' caricature, hence the name, however the model can be trained to generate any desired output given the relevant training material. NOTE: The currently included generator model, trained off 97 different input/output image pairs, is not able to generate sufficiently acceptable output due to a lack of training data.

File Structure:
- dataset (Folder): Contains training data in the form of .jpg, .png and .webp image files, labelled in pairs of "imageX_input", "imageX_output", as well as the image outputs of the generator.
- saved_model (Folder): Contains the AI generator model
- dataset_loader.py: Contains functions which load the data from the dataset folder into a suitable dataset for the AI model to be trained on
- doro_generator.py: Contains the main generator (Adversarial GAN) model and its training program, applying various techniques to improve its generation quality such as Self-Attention, regularization and learning rate optimizers. Run this file to train the generator model on the provided dataset and save it in the saved_model folder.
- generate_doro.py: Used to operate the generator, accepting a single input image and utilising the trained generator model to attempt to produce a corresponding output image. Run this file to operate the generator and save the produced output image in the dataset folder.
- image_formatter.py: Contains functions which format files such that they conform to a uniform format accepted by the generator model - specifically a 256x256 .png file.
