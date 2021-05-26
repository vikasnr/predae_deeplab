            # -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:15:10 2021

@author: aluque

We want to reproduce the model of PredAE as descrined in
Jan-Aike Bolte, Andreas Bar, Daniel Lipinski and Tim Fingscheidt
Towards Corner Case Detection for Autonomous Driving.
So, unless otherwise stated, we will refer to this paper

List of things to check/parameters to adjust before refactoring the code:
    - Number of parameters. We do NOT have the same number of parameters that
    they list in Table II. We are doing something very similar but not the same
    
    - For the moment we start with a learning rate of order 1e-3 rather than
    1e-7 and the use in the paper. This is because there start with a
    pretrained model and we start from scratch. We will decrease the rate later
    
On the environment:
    tensorflow 2.4.1
    numpy 1.19.5
    keras 2.4.3
    
"""

import numpy as np
import tensorflow as tf
import pathlib

# from tensorflow.keras import mixed_precision
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

###############################################################################
# Global variables

nt = 9        # Number of images used to predict the next one
H = 1024      # Dimensions of the original image
W = 2048
HP = 256      # Dimensions used by PredAE (will be 256x512 later)
WP = 512
HD = 512      # Dimensions used by DeepLabv3
WD = 1024

if HP%8==0 & WP%8==0:
    Hdis=int(HP/8)
    Wdis=int(WP/8)
    print("Discriminator output shape: (None,{},{},1)".format(Hdis,Wdis))
else:
    raise ValueError("Dimensions used in PredAE are not compatible")

gen_lr = 1e-3 # Learning rate to train the generator
dis_lr = 1e-3 # Learning rate to train the discriminator
gan_lr = 1e-3 # Learning rate to train the gan

L2reg = 1e-3  # L2 regularization parameter 

BS = 3        # Batch size for each training

###############################################################################
# Loading the data using tf.data.Dataset

def load_image(image_file,deeplab=False):
    """
    This function loads a single image and preprocesses it according to the 
    requirements of PredAE

    Parameters
    ----------
    image_file : string or tensor
        path for the image

    Returns
    -------
    image : tensor
        preprocessed imaged. The image has a single channel (gray scale),
        is scaled to (0,1) and reduced to fit the required size

    """
    HI = HP
    WI = WP
    if deeplab:
        HI = HD
        WI = WD
        
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=1)
    
    image = tf.cast(image, tf.float32)
    image = image/255.0
    image = tf.image.resize(image, 
                            [HI, WI],
                            method=tf.image.ResizeMethod.AREA)
    
    return image

@tf.autograph.experimental.do_not_convert
def load_sequence_images(image_file):
    """
    This function loads a sequence of consecutive nt+1 images from a directory.
    The first nt images are concatenated into a tensor of shape HPxWPxnt, 
    where the entry [i,j,k] stands for the (i,j)-pixel of the image k-th image. 
    The nt+1-th image is returned separately, since it is the image that we
    want to predict.
    
    WARNING: the routina is problem dependent, in the sense that it relies on
    the names of the files in Cityscapes. 

    Parameters
    ----------
    image_file : string or eager tensor
        path for the first image in the sequence

    Returns
    -------
    sequence : tensor
        Sequence of nt consecutive images, with shape [HP, WP, nt] 
    next_image : tensor
        nt+1 image that we want to predict, with shape [HP, WP, 1]
        
    Remark: Notice that we need to recover the "string" that contains tha path
        in order to find the consecutive images.
        
    """
    current_file = image_file.numpy().decode('utf-8')
    # We first get the number corresponding to the initial image
    file_split = current_file.split("_")
    current_number = int(file_split[-2])
    
    img_seq = []
    # We load and preprocess the sequence of images 
    for number in range(current_number, current_number+nt):
        img_seq.append(load_image(current_file))
        
        # Get the name of the next file
        file_split[-2]=f'{number:06d}'
        next_file = "_".join(file_split)
        current_file = next_file
        
    sequence = tf.concat(img_seq, axis=2)
    next_image = load_image(current_file,deeplab=False)
    
    return sequence, next_image, current_file

def set_dataset_iterator(data_dirs):
    """
    This function creates a tensorflow prefetch dataset iterable to load the
    sequences of images at the same time that they are required during the
    training. For example, imagine that we have two directoties with images

    dir1 contains im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12
    dir2 contains im25, im26, im27, im28, im29, im30, im31, im32, im33, im34, im35

    We will be able to create 5 pairs of sequence and predicted image:

        (im1, im2, im3, im4, im5, im6, im7, im8, im9) --> im10
        (im2, im3, im4, im5, im6, im7, im8, im9, im10) --> im11
        (im3, im4, im5, im6, im7, im8, im9, im10, im11) --> im12
        (im25, im26, im27, im28, im29, im30, im31, im32, im33) --> im34
        (im26, im27, im28, im29, im30, im31, im32, im33, im34) --> im35

    Notice that before returning the iterator, these examples are shuffled.   

    Parameters
    ----------
    data_dirs : list
        This is a list of directories (path). Each directory contains a 
        sequence of consecutive images forming a video. 

    Returns
    -------
    train_ds : tensorflow iterable that returns batches of data
        Each batch contains BS examples of pairs of sequences of nt images and
        the corresponding image that we want to predict.
        Notice that train_ds is not an iterator, so next(train_ds) is not
        defined directly

    """
    list_of_files = []
    for data_dir in data_dirs:
        data_dir = pathlib.Path(data_dir)
        local_list_of_files = [str(file) 
                               for file in list(data_dir.glob('*.png'))]
        local_list_of_files.sort()
        local_list_of_files = local_list_of_files[:-nt]
        list_of_files = list_of_files + local_list_of_files
    
    train_ds = tf.data.Dataset.list_files(list_of_files, shuffle=True)
    
    train_ds = train_ds.map(
                            lambda x: tf.py_function(
                                                     load_sequence_images,
                                                     [x],
                                                     [tf.float32, tf.float32,tf.string]
                                                     ),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE
                            )
    
    train_ds = train_ds.batch(batch_size = BS).prefetch(1)
    
    return train_ds

###############################################################################
# Define the models

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.optimizers import Adagrad, SGD

def create_generator(summary=True, file=None):
    """
    Create or load a generator model. For convenience, we compile this model
    with its own loss function, in case that we want to train it separately
    from the GAN architecture

    Parameters
    ----------
    summary : boolean
        If True we print the summary of the model after loading it.
        The default is True.
    file : string or None
        If no string is given, we define the model according Fig 4 (see paper).
        If string is the path of a h5 file (pretrained model), we load it.

    Returns
    -------
    generator : keras Sequential model
        - The input of this model is a sequence of frames. 
          Input shape is (None,HP,WP,nt).
        - The output of the model is the prediction of the next frame. 
          Output shape is (None,HP,WP,1)
        - The model is compiled with MSE loss and Adagrad, using gen_lr 
          learning rate
          
    Remark: Notice that the last activation is provided in a separate layer, so
        that we can run this model using mixed_precision
        
    """
    if file==None:
        generator = Sequential([
            Conv2D(filters=512, kernel_size=5, strides=1, activation='relu', 
                   padding="SAME", input_shape=[HP, WP, nt], 
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=256, kernel_size=5, strides=1, activation='relu',
                   padding="SAME", 
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', 
                   padding="SAME", 
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            UpSampling2D(size=2, interpolation='nearest'),
            Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', 
                   padding="SAME", 
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            UpSampling2D(size=2, interpolation='nearest'),
            Conv2D(filters=512, kernel_size=5, strides=1, activation='relu', 
                   padding="SAME",
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            Conv2D(filters=1, kernel_size=5, strides=1, activation=None, 
                   padding="SAME",
                   kernel_regularizer = tf.keras.regularizers.l2(L2reg)),
            Activation('relu', dtype='float32'),
        ])
        
    else:
        generator = load_model(file)
        
    generator.compile(loss="mean_squared_error", optimizer=Adagrad(lr=gen_lr))
    
    if summary==True:
        generator.summary()
    
    return generator

def create_discriminator(summary=True, file=None):
    """
    Create or load a discriminator model. We compile this model with its own
    loss function. This loss function will be called during the training of
    the discriminator inside the GAN

    Parameters
    ----------
    summary : boolean
        If True we print the summary of the model after loading it.
        The default is True.
    file : string or None
        If no string is given, we define the model according Fig 5 (see paper).
        If string is the path of a h5 file (pretrained model), we load it.

    Returns
    -------
    discriminator : keras Sequential model
        - The input of this model is an image. 
          Input shape is (None,HP,WP,1).
        - The output of the model is a matrix of sigmoid activation that gives
          local information on the probability that the image is real
          Output shape is (None,Hdis,Wdis,1)
        - Following the model should be compiled with BCE loss and SGD, 
          using dis_lr learning rate. In this version of the code we use
          Adagrad
        - By default, the discriminator weights are set to non-trainable

    Remark: Notice that the last activation is provided in a separate layer, so
        that we can run this model using mixed_precision

    """
    if file==None:
        discriminator = Sequential([
            # We start reproducing the structure of the Encoder
            Conv2D(filters=512, kernel_size=5, strides=1, activation='relu',
                   padding="SAME", input_shape=[HP, WP, 1]),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=256, kernel_size=5, strides=1, activation='relu',
                   padding="SAME"),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', 
                   padding="SAME"),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=1, kernel_size=2, strides=1, activation=None, 
                   padding="SAME"),
            Activation('sigmoid', dtype='float32'),
        ])
        
    else:
        discriminator = load_model(file)

    discriminator.compile(loss="binary_crossentropy", optimizer=Adagrad(lr=dis_lr))
    discriminator.trainable = False
    
    if summary==True:
        discriminator.summary()
        
    return discriminator

def create_gan(generator, discriminator, summary=True):
    """
    Create a GAN model

    Parameters
    ----------
    generator : keras Sequential model
    discriminator : keras Sequential model
    summary : boolean
        If True we print the summary of the model after loading it.
        The default is True.

    Returns
    -------
    discriminator : keras Model model
        - The input of this model is a sequence of frames. 
          Input shape is (None,HP,WP,nt).
        - The output of the model is a tuple consisting on the prediction of 
          the next frame (generator output) a the matrix of local probabilities
          (discriminator output)
          Output shape is [(None,HP,WP,1), (None,Hdis,Wdis,1)]
        - The model is compiled with adversarial loss (that is MSE+0.25*BCE)
          and SGD, using gan_lr learning rate.

    """
    discriminator.trainable=False
    
    gan_input  = generator.input           # Input is (x_1, x_2, ..., x_{n-1})
    output_gen = generator(gan_input)      # \hat x_n = Gen(x_n)
    output_dis = discriminator(output_gen) # \hat y_n = Dis(Gen(x_n))
    
    gan = Model(inputs=gan_input, outputs=[output_gen, output_dis])
    
    # For output_gen we use MSE comparing x_n with \hat x_n
    # For output_dist we use BCE comparing Dis(x_n) with 1
    
    # Loss = MSE+0.25*BCE
    gan.compile(loss=['mean_squared_error','binary_crossentropy'],
                loss_weights=[1.0, 0.25],    
                optimizer=SGD(lr=gan_lr))
    
    if summary==True:
        gan.summary()

    return gan

###############################################################################
# Training strategies

def train_generator(generator, dataset, n_epochs=10):
    """
    Train the generator alone for some epocs

    Parameters
    ----------
    generator : keras Sequential model
    dataset : prefetch dataset iterable
        Can be iterated to take batches of BS examples consisting on sequences
        of images and the next image to be predicted
    n_epochs : int
        Number of epochs. The default is 10.

    Remark: We train on batches just for symmetry in the codes. We could use
      the fit method equivalently.

    """
    print("--- We train the generator alone for {} epochs".format(n_epochs))
    for epoch in range(n_epochs):
        train_loss = 0.
        for sequence, next_image in dataset:
            train_loss += generator.train_on_batch(x=sequence, y=next_image)

        print("epoch {}, GEN_loss {}".format(epoch+1, train_loss))
        
    return str("Training completed!")
        
def train_gan_discriminator_only(gan, dataset, n_epochs=10):
    """
    Train the discriminator alone for some epocs

    Parameters
    ----------
    gan : keras Model model
    dataset : prefetch dataset iterable
        Can be iterated to take batches of BS examples consisting on sequences
        of images and the next image to be predicted
    n_epochs : int
        Number of epochs. The default is 10.
        
    Remark: Notice that we need the full gan as input, since we need to use the
      generator to create predicted frames

    """
    # Get the models
    gan_input, generator, discriminator = gan.layers
    discriminator.trainable = True
    
    print("--- We train the discriminator alone for {} epochs".format(n_epochs))
    for epoch in range(n_epochs):
        train_loss = 0.
        for sequence, next_image in dataset:
            # First we use the real images (labelled as 1's)
            y1 = tf.constant(np.ones((next_image.shape[0],Hdis,Wdis,1)))
            train_loss += discriminator.train_on_batch(x=next_image, y=y1)

            # Then we use the generated imafes (labelled as 0's)
            pred_image = generator.predict_on_batch(x=sequence)
            y0 = tf.constant(np.zeros((next_image.shape[0],Hdis,Wdis,1)))
            train_loss += discriminator.train_on_batch(x=pred_image, y=y0)
            
        print("epoch {}, DIS_loss {}".format(epoch+1, train_loss))
        
    discriminator.trainable = False
    return str("Training completed!")

def train_gan(gan, dataset, n_epochs=10):
    """
    Train the gan for some epochs: In even batches we train the weights of the
    discriminator and in odd batches we train the weights of the generator.

    Parameters
    ----------
    gan : keras Model model
    dataset : prefetch dataset iterable
        Can be iterated to take batches of BS examples consisting on sequences
        of images and the next image to be predicted
    n_epochs : int
        Number of epochs. The default is 10.

    Remark: Notice that when we train the models using this function, the total
      examples used in each epoch is halved with respect to the functions
      train_generator and train_gan_discriminator_only. As a consequence, the
      value of the loss is halved
      
    """
    # Get the models
    gan_input, generator, discriminator = gan.layers
    print("--- We train the gan for {} epochs".format(n_epochs))
    for epoch in range(n_epochs):
        gan_train_loss = 0.
        dis_train_loss = 0.
        for bn, (sequence, next_image) in enumerate(dataset):
            if bn%2==0: # Train the discriminator
                discriminator.trainable = True
                
                # First we use the real images (labelled as 1's)
                y1 = tf.constant(np.ones((next_image.shape[0],Hdis,Wdis,1)))
                dis_train_loss += discriminator.train_on_batch(x=next_image, y=y1)
    
                # Then we use the generated imafes (labelled as 0's)
                pred_image = generator.predict_on_batch(x=sequence)
                y0 = tf.constant(np.zeros((next_image.shape[0],Hdis,Wdis,1)))
                dis_train_loss += discriminator.train_on_batch(x=pred_image, y=y0)

            else: # Train the generator
                discriminator.trainable = False

                y1 = tf.constant(np.ones((next_image.shape[0],Hdis,Wdis,1)))
                gan_train_loss += gan.train_on_batch(sequence, [next_image, y1])[0]
         
        # For the sake of comparison, we also plot the loss associated to the 
        # generator (MSE alone)
        print("epoch {}, GAN_loss {}, DIS_loss {}, GEN_loss {}".format(
            epoch+1, 
            gan_train_loss, 
            dis_train_loss,
            gan_train_loss-0.25*dis_train_loss))

    discriminator.trainable = False
    return str("Training completed!")