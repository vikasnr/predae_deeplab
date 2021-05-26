# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:46:59 2021

@author: aluque

This script loads pre-trained generator and discriminator models from h5 file 
and evaluates the examples in test_dirs, to test how well the discriminator
distinguishes real and generated images. Since we are not using the
discriminator to "classify" but to evaluate the loss function, we print the
predicted probabilities, to see how much both types of images are distinguished
"""

from predAE import set_dataset_iterator
from predAE import create_generator, create_discriminator
from datainfo import test_dirs, models_dir
import numpy as np

# Prepare the dataset (that we want to test)
test_ds=set_dataset_iterator(test_dirs)

gen_file = models_dir + "generator.h5"
dis_file = models_dir + "discriminator.h5"

#Create the models
gen_model = create_generator(summary=False, file=gen_file)
dis_model = create_discriminator(summary=False, file=dis_file)

# Iterate along the dataset and predict the frames
for sequence, next_image in test_ds:
    p1=dis_model.predict_on_batch(next_image)
    for bn in range(next_image.shape[0]):
        print("real", np.mean(p1[bn]))

    pred_image = gen_model.predict_on_batch(sequence)
    p0=dis_model.predict_on_batch(pred_image)
    for bn in range(next_image.shape[0]):
        print("pred", np.mean(p0[bn]))
    