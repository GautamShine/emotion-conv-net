#!/usr/bin/env python

###############################################################################
# OpenCV + Caffe
# Uses OpenCV for face detection and cropping to serve as inputs to Caffe.
# Uses a Caffe VGG_S net from EmotiW'15 [1] to classify emotion.
# Authors: Gautam Shine, Dan Duncan
#
# [1] Gil Levi and Tal Hassner, Emotion Recognition in the Wild via
# Convolutional Neural Networks and Mapped Binary Patterns, Proc. ACM
# International Conference on Multimodal Interaction (ICMI), Seattle, Nov. 2015
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

from caffe_functions import *
from opencv_functions import *
from utility_functions import *

### USER-SPECIFIED VARIABLES: ###

# List your dataset root directories here:
dirCKPlus = 'datasets/CK_Plus'

# Select which dataset to use (case insensitive):
dataset = 'ckplus'

# Flags:
cropFlag = True # False disables image cropping

### START SCRIPT: ###

# Set up inputs
dir = dirCKPlus
color = False
single_face = True

# Clean up and discard anything from the last run
dirCrop = dir + '/cropped'
rmdir(dirCrop)

# Master list of categories for EmotitW network
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
suffixes   = ['AN',      'DI',       'FE',    'HA',      'NE',        'SA',   'SU']

# Load dataset image list
input_list, labels = importDataset(dir, dataset, categories)

# Perform detection and cropping if desired (and it should be desired)
mkdir(dirCrop)
input_list = faceCrop(dirCrop, input_list, color, single_face)

# Print outs
print input_list
print labels

# Rename all files
for i in range(len(input_list)):
  # Get file info
  filename = input_list[i]
  lab = labels[i]
  labText = suffixes[lab]

  # Generate new filename
  fn = filename.split('.')
  out = fn[0] + '.' + labText + '.' + fn[1]

  # Rename file
  os.rename(filename,out)


  
