# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:46:59 2021

@author: aluque
"""

NE = 2

import tensorflow as tf
from predAE import set_dataset_iterator
from predAE import create_generator
from predAE import create_discriminator
from predAE import create_gan
from predAE import train_gan
from datainfo import data_dirs, models_dir

#Prepare the dataset
train_ds=set_dataset_iterator(data_dirs)

gen_file = models_dir + "generator.h5"
dis_file = models_dir + "discriminator.h5"

#Create the models (WARNING: We reset both generator and discriminator!)
gen_model = create_generator(summary=False)
dis_model = create_discriminator(summary=False)
gan_model = create_gan(gen_model, dis_model, summary=False)

train_gan(gan_model, train_ds, n_epochs=NE)       

#Save/uptade the weights in h5 format
inputs, gen_model, dis_model = gan_model.layers
tf.keras.models.save_model(gen_model, gen_file, save_format='h5')
tf.keras.models.save_model(dis_model, dis_file, save_format='h5')
