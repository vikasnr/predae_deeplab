# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:46:59 2021

@author: aluque

This script creates a generator model and trains it from scratch for NE
epochs. The trained model is saved in a h5 file
"""

NE = 10

import tensorflow as tf
from predAE import set_dataset_iterator
from predAE import create_generator
from predAE import train_generator
from datainfo import data_dirs, models_dir

#Prepare the dataset
train_ds=set_dataset_iterator(data_dirs)

gen_file = models_dir + "generator.h5"

#Create the model
gen_model = create_generator(summary=False)

train_generator(gen_model, train_ds, n_epochs=NE)       

#Save/uptade the weights in h5 format
tf.keras.models.save_model(gen_model, gen_file, save_format='h5')
