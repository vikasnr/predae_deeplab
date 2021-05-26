# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:23:33 2021

@author: BGH52648
"""

#DEEPLAB
import sys

from datasets import DatasetVal_2 # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

from deeplabv3 import DeepLabV3

from utils import label_img_to_color

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import cv2
import re
import os


H = 1024 # Dimensions of the image
W = 2048
HD = 512 # Dimensions used by DeepLabv3
WD = 1024
HP = 128 # Dimensions used by PredNet
WP = 160

def error_score_pred(e, mt):    
    Srel = np.array(range(11,19))
    
    H,W = mt.shape
    
    try:
        for i in range(H):
            for j in range(W):
                if(mt[i,j] not in Srel):
                    e[i][j] = 0
        ew = e                    
        es = 0
        for i in range(H):
            for j in range(W):
                ew[i,j] = ew[i,j] * (1- ((H-i-1)/(H-1)))
                es = es + ((e[i][j]*e[i][j]) * (1 -  ((H-i-1)/(H-1))))
        
        return es,ew,e
    except IndexError as e :
        print(e)
        print(i,j)



#%%

from deeplabv3 import DeepLabV3
# from wideresnet import WideResNet
import torch.nn.functional as F
import torch.nn as nn

def load_deeplab_model(model_id,model_path,project_dir="D:/Altran/Edge/towards corner case/deeplabv3"):
    model_id = '13_2_2_2'
    network = DeepLabV3(model_id, project_dir=project_dir)
    network.load_state_dict(torch.load(model_path))
    return network

#Resnet
model_id = '13_2_2_2'
model_path = "D:/Altran/Edge/towards corner case/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth"

# wideresnet
# model_id = 'id_6_epoch_10_all_bt_8'
# model_path = "D:/Altran/Edge/towards corner case/deeplabv3/pretrained_models/id_6_epoch_10_all_bt_8.pth"

network = load_deeplab_model(model_id,model_path)

network.eval()

with open("D:/Altran/Edge/towards corner case/deeplabv3/data/cityscapes/meta/class_weights.pkl", "rb") as file:
    class_weights = np.array(pickle.load(file))
    
     
class_weights = torch.from_numpy(class_weights)
class_weights = Variable(class_weights.type(torch.FloatTensor))
    
# loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
# Semantic Segmentation
def semantic_segmentation(predicted_path):
    batch_size = 1
    val_dataset = DatasetVal_2(image_path=predicted_path,frame_id = 'next')
    num_val_batches = int(len(val_dataset)/batch_size)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             shuffle=False,
                                             num_workers=0)
    batch_losses = []
    
    for step, (imgs) in enumerate(val_loader):
        with torch.no_grad():
            imgs = Variable(imgs) # (shape: (batch_size, 3, img_h, img_w))
            output = network(imgs) # (shape: (batch_size, num_classes, img_h, img_w))
 
            output = output.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
            pred_label_imgs = np.argmax(output, axis=1) # (shape: (batch_size, img_h, img_w))
            pred_label_imgs = pred_label_imgs.astype(np.uint8)

            return pred_label_imgs[0]


#%%
#PRED-AE
import sys
from predAE import set_dataset_iterator
from predAE import create_generator
from datainfo import test_dirs, models_dir
import matplotlib.pyplot as plt

plot = True

# Prepare the dataset (that we want to test)
test_ds=set_dataset_iterator(test_dirs)

# Create the model
gen_file = models_dir + "generator.h5"
gen_model = create_generator(summary=False, file=gen_file)

# Iterate along the dataset and predict the frames
for sequence, next_image, next_image_path in test_ds:
    pred_images = gen_model.predict_on_batch(sequence)
    for bn in range(pred_images.shape[0]):
     
        next_im_path = next_image_path.numpy()[bn].decode("utf-8")
        
        sem_seg = semantic_segmentation(next_im_path)
        
        pred_error =  pred_images[bn][:,:,0] - next_image[bn].numpy()[:,:,0]
        
        pred_errorR = cv2.resize(pred_error, (WD, HD), interpolation=cv2.INTER_AREA)
        eps_score,wE,eR = error_score_pred(pred_errorR,sem_seg)

        #if plot=true then plot actual,predicted and semantic segmentation
        if plot:
            
            pred_label_img_color = label_img_to_color(sem_seg)
            overlayed_img = 0.35*(cv2.resize(cv2.imread(next_im_path,-1), (WD, HD), interpolation=cv2.INTER_AREA)) + 0.65*pred_label_img_color
            overlayed_img = overlayed_img.astype(np.uint8)
            plt.imshow(overlayed_img)
            plt.show()
            
            plt.imshow(next_image[bn,:,:,0]) # In Colab
            plt.show()
            
    
            plt.imshow(pred_images[bn,:,:,0]) # In Colab
            plt.show()
        
        

