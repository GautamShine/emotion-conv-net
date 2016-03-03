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
dirJaffe = 'datasets/jaffe'
dirCKPlus = None  # TODO
# dirOther ... TODO: allow any generic directory of pictures

# Select which dataset to use (case insensitive):
dataset = 'jaffe';

# Flags:
cropFlag = True # False disables image cropping

### START SCRIPT: ###

# Set up inputs
dir = None
if dataset.lower() == 'jaffe':
    dir = dirJaffe;
else:
    print 'Error - Unsupported dataset: ' + dataset
    sys.exit(0)

# Clean up and discard anything from the last run
dirCrop = dir + '/cropped';
rmdir(dirCrop);

# Master list of categories for EmotitW network
categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

# Load dataset image list
input_list, labels = importDataset(dir, dataset, categories)

# Load Haar cascade files containing features
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)

# Perform detection and cropping if desired (and it should be desired)
if cropFlag:
    mkdir(dirCrop)
    input_list = faceCrop(dirCrop, input_list, faceCascade, color=False)

# Perform classification
classify_emotions(input_list, categories, labels, plot_neurons=False, plot_confusion=True)
