# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:46:59 2021

@author: aluque

This script loads a pre-trained generator model from h5 file and evaluates the
examples in test_dirs, to test how well the predictions look
"""

from predAE import set_dataset_iterator
from predAE import create_generator
from datainfo import test_dirs, models_dir
import matplotlib.pyplot as plt

# Prepare the dataset (that we want to test)
test_ds=set_dataset_iterator(test_dirs)

# Create the model
gen_file = models_dir + "generator.h5"
gen_model = create_generator(summary=False, file=gen_file)

# Iterate along the dataset and predict the frames
for sequence, next_image in test_ds:
    pred_image = gen_model.predict_on_batch(sequence)
    for bn in range(pred_image.shape[0]):
        # Plot the next image of the sequence (the real one)
        # plt.imshow(next_image[bn])       # In local computer
        plt.imshow(next_image[bn,:,:,0]) # In Colab
        plt.show()
        # Plot the predicted image by the generator
        # plt.imshow(pred_image[bn])       # In local computer
        plt.imshow(pred_image[bn,:,:,0]) # In Colab
        plt.show()
