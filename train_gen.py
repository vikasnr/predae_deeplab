# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:46:59 2021

@author: aluque

This script loads a pre-trained generator model from h5 file and trains it for
NE epochs. The trained model is saved in the same h5 file.
"""

NE = 2

import tensorflow as tf
from predAE import set_dataset_iterator
from predAE import create_generator
from predAE import train_generator
from datainfo import data_dirs, models_dir

#Prepare the dataset
train_ds=set_dataset_iterator(data_dirs)

#Create the model
gen_file = models_dir + "generator.h5"

gen_model = create_generator(summary=False, file=gen_file)

train_generator(gen_model, train_ds, n_epochs=NE)       

#Save/uptade the weights in h5 format
tf.keras.models.save_model(gen_model, gen_file, save_format='h5')
