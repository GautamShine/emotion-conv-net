###############################################################################
# Caffe VGG_S net emotion classification
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

from utility_functions import *

# Display an image (input is numpy array)
def showimage(img):
    if img.ndim == 3:
        img = img[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(img,vmin=0, vmax=0.2)

# Display network activations
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # Force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # Tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    showimage(data)

# Plot the last image and conv1 layer's weights and responses
def plot_layer(input_image, VGG_S_Net, layer):
    plt.figure(1)
    _ = plt.imshow(input_image)

    plt.figure(2)
    filters = VGG_S_Net.params[layer][0].data
    vis_square(filters.transpose(0, 2, 3, 1))

    plt.figure(3)
    feat = VGG_S_Net.blobs[layer].data[0]
    vis_square(feat)

    plt.show(block=False)

# RGB dimension swap + resize
# Output should be 3x256x256 for VGG_S net regardless of input shape
def mod_dim(img, x=256, y=256, c=3):
    # Resize only if necessary:
    if not np.array_equal(img.shape,[c,x,y]):
        resized = caffe.io.resize_image(img, (x,y,c)) # (256, 256, 3)
        rearranged = np.swapaxes(np.swapaxes(resized, 1, 2), 0, 1) # (3,256,256)
    else:
        rearranged = img

    return rearranged

# Calculate mean over list of filenames
def compute_mean(input_list, plot_mean=False):
    # If no data supplied, use mean supplied with pretrained model
    if len(input_list) == 0:
        net_root = '.'
        net_dir = 'VGG_S_rgb'
        mean_filename=os.path.join(net_root, net_dir, 'mean.binaryproto')
        proto_data = open(mean_filename, "rb").read()
        a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
        mean  = caffe.io.blobproto_to_array(a)[0]
    else:
        x,y,c = 256,256,3
        mean = np.zeros((c, x, y))
        for img_file in input_list:
            img = caffe.io.load_image(img_file)
            img = mod_dim(img, x, y, c)
            mean += img
        mean /= len(input_list)

        # Plot the mean image if desired:
        if plot_mean:
            plt.imshow(np.swapaxes(np.swapaxes(mean, 0, 1), 1, 2))
            plt.show()
    return mean

# Return VGG_S_Net from mean image and optional network type
def make_net(mean=None, net_dir='VGG_S_rgb'):
    # net_dir specifies type of network 
    # Options are: (rgb, lbp, cyclic_lbp, cyclic_lbp_5, cyclic_lbp_10)

    caffe_root = '/home/gshine/Data/Caffe'
    sys.path.insert(0, caffe_root + 'python')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    net_root = 'models'

    net_pretrained = os.path.join(net_root, net_dir, 'EmotiW_VGG_S.caffemodel')
    net_model_file = os.path.join(net_root, net_dir, 'deploy.prototxt')
    VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return VGG_S_Net

# Load a minibatch of images
def load_minibatch(input_list, color, labels, start,num):
    # Enforce maximum on start
    start = max(0,start)

    # Enforce minimum on end
    end = start + num
    end = min(len(input_list), end)

    # Isolate files
    files = input_list[start:end]

    images = []
    for file in files:
        img = caffe.io.load_image(file, color)

        # Handle incorrect image dims for uncropped images
        # TODO: Get uncropped images to import correctly
        if img.shape[0] == 3:
            img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        images.append(img)

    labelsReduced = labels[start:end]
    return images, labelsReduced

# Classify all images in a list of image file names
# No return value, but can display outputs if desired
def classify_emotions(input_list, color, categories, labels, plot_neurons, plot_confusion):
    # Compute mean
    mean = compute_mean(input_list)

    # Create VGG_S net with mean
    VGG_S_Net = make_net(mean)

    # Classify images in directory
    conf_mat = [] # tuples to be passed to confusion matrix generator

    numImages = len(input_list)

    miniBatch = False
    if miniBatch:
        i = 0
        batchSize = 500
        while i < numImages:
            images,labelsReduced = load_minibatch(input_list, color, labels, i, batchSize)
            print 'Batch of  ' + str(len(images)) + '  images.'
            prediction = VGG_S_Net.predict(images, oversample=False)

            for j in range(len(prediction)):
                pred = prediction[j]
                lab = labelsReduced[j]

                # Append (label, prediction) tuple to confusion matrix list
                conf_mat.append((lab, pred.argmax()))

                # Print results as Filename: Prediction
                print(input_list[i+j].split('/')[-1]+': '+categories[prediction.argmax()])

            i += batchSize
    else:
        for i in range(numImages):
            img_file = input_list[i]
            label = labels[i]

            print('File name: ', img_file)
            input_image = caffe.io.load_image(img_file)

            # Handle incorrect image dims for uncropped images
            # TODO: Get uncropped images to import correctly
            if input_image.shape[0] == 3:
                input_image = np.swapaxes(np.swapaxes(input_image, 0, 1), 1, 2)
            prediction = VGG_S_Net.predict([input_image], oversample=False)

            # Append (label, prediction) tuple to confusion matrix list
            conf_mat.append((label, prediction.argmax()))

            # Print results as Filename: Prediction
            print(img_file.split('/')[-1]+': '+categories[prediction.argmax()])

    if plot_neurons:
        layer = 'conv1'
        plot_layer(input_image, VGG_S_Net, layer)

    # Generates confusion matrix and calculates accuracy
    confusion_matrix(conf_mat, categories, plot_confusion)
