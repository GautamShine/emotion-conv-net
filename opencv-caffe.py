#!/usr/bin/env python

###############################################################################
# OpenCV + Caffe
# Uses OpenCV for face detection and cropping to serve as inputs to Caffe.
# Uses a Caffe VGG_S net from EmotiW'15 [1] to classify emotion.
# Author: Gautam Shine
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

###############################################################################
# Caffe VGG_S net emotion classification
###############################################################################

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

# RGB dimension swap + resize
# Output should be 3x256x256 regardless of input shape
def mod_dim(img, x=256, y=256, c=3):
    #print 'Image shape:'
    #print img.shape;

    # Resize only if necessary:
    if not np.array_equal(img.shape,[c,x,y]):
        resized = caffe.io.resize_image(img, (x,y,c)) # (256, 256, 3)
        rearranged = np.swapaxes(np.swapaxes(resized, 1, 2), 0, 1) # (3, 256, 256)

    else:
        rearranged = img;


    #print 'Image swapped shape:'
    #print rearranged.shape

    #sys.exit(0)
    return rearranged

# Calculate mean over list of filenames
def compute_mean(input_list, plot_mean = False):
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
    #net_dir = 'VGG_S_rgb' 
    

    caffe_root = '/Users/Dan/Development/caffe/'
    sys.path.insert(0, caffe_root + 'python')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

    net_root = '.'
    
    net_pretrained = os.path.join(net_root, net_dir, 'EmotiW_VGG_S.caffemodel')
    net_model_file = os.path.join(net_root, net_dir, 'deploy.prototxt')
    VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return VGG_S_Net, categories

# Plot confusion matrix
def plot_confusion_matrix(cm, names=None, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Add labels to confusion matrix:
    if names is None:
        names = range(cm.shape[0]);

    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    
    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')
    plt.show()

# Generate confusion matrix for Jaffe:
# results = list of tuples of (correct label, predicted label)
#         e.g. [ ('HA', 3) ]
# categories = list of category names
# Returns confusion matrix where rows are correct labels and columns are predictions
def confusion(results, categories, plotConfusion=False):
    """
    map_categories = {
        'HA': categories.index('Happy'),
        'SA': categories.index('Sad'),
        'NE': categories.index('Neutral'),
        'AN': categories.index('Angry'),
        'FE': categories.index('Fear'),
        'DI': categories.index('Disgust'),
        'SU': categories.index('Surprise')
    }
    """

    # Empty confusion matrix:
    matrix = np.zeros((7,7))

    # Iterate over all labels and populate matrix
    for label, pred in results:
        matrix[label,pred] += 1;
        #matrix[map_categories[label],pred] += 1
    
    # Print matrix and percent accuracy
    accuracy = float(np.trace(matrix))/len(results)
    print('Confusion Matrix: ')
    print(matrix)
    print 'Accuracy: ' +  str(accuracy*100) + '%'

    # Plot the confusion matrix
    if plotConfusion:
        plot_confusion_matrix(matrix, categories)


# Classify all images in a list of image file names:
# No return value, but can display outputs if desired
def classify_emotions(input_list, labels, show_confusion, show_faces, show_neurons):
    
    # Master categories list:
    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'];


    # Compute mean
    mean = compute_mean(input_list)
    #mean = None; # Cancel out mean
    
    # Create VGG_S net with mean
    VGG_S_Net, categories = make_net(mean)

    # Classify images in directory
    conf_mat = [] # confusion matrix (holds list of tuples to be passed to confusion matrix generator)


    numImages = len(input_list);

    for i in range(numImages):

        img_file = input_list[i]
        label = labels[i]

        print 'File name: ' + img_file;
        input_image = caffe.io.load_image(img_file)  # 3x256x256 for no crop, 145x145x3 for crop

        # Handle incorrect image dims for uncropped images
        # TODO: Get uncropped images to import correctly
        if input_image.shape[0] == 3:
            input_image = np.swapaxes(np.swapaxes(input_image, 0, 1), 1, 2) # (256, 256, 3)
        
        #print 'Input Image shape (line 189):'
        #print input_image.shape; 
        #sys.exit(0)

        # TODO: test larger crops + oversampling
        prediction = VGG_S_Net.predict([input_image], oversample=False)

        # Append (label, prediction) tuple to confusion matrix list:
        #if show_confusion:
        #label = img_file.split('.')[1][0:2]
        # conf_mat now has an integer label and an integer prediction:
        conf_mat.append((label, prediction.argmax()))

        # Print results as:  Filename: Prediction
        print(img_file.split('/')[-1] + ': ' + categories[prediction.argmax()])
        
        # TODO: Prediction currently breaks after 1st face if plotting is on
        if show_faces:
            plt.figure(1)
            _ = plt.imshow(input_image)

        if show_neurons:
            plt.figure(2)
            filters = VGG_S_Net.params['conv1'][0].data
            vis_square(filters.transpose(0, 2, 3, 1))

            plt.figure(3)
            feat = VGG_S_Net.blobs['conv1'].data[0]
            vis_square(feat)

        if show_faces or show_neurons:
            plt.show(block=False)
            time.sleep(1)
            plt.close('all')

    # Generates confusion matrix and calculates accuracy
    # Only plots confusion matrix if show_confusion==True
    confusion(conf_mat, categories, show_confusion)

# Get entire dataset:
# Inputs: Dataset root directory; optional dataset name
# Returns: List of all image file paths; list of correct labels for each image
# TODO: Expand to be able to import multiple datasets
def importDataset(dir,dataset='jaffe'):
    # Master list of categories for EmotitW network:
    # Now global variable
    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise'];

    imgList = None;
    labels = None;
    
    # Datset-specific import rules:
    if dataset.lower() == 'jaffe':
        # Get Jaffe file names
        imgList = glob.glob(dir+'/*')

        # Get Jaffe labels
        jaffe_categories_map = {
            'HA': categories.index('Happy'),
            'SA': categories.index('Sad'),
            'NE': categories.index('Neutral'),
            'AN': categories.index('Angry'),
            'FE': categories.index('Fear'),
            'DI': categories.index('Disgust'),
            'SU': categories.index('Surprise')
            }

        labels = [];

        for img in imgList:
            key = img.split('.')[1][0:2]
            labels.append(jaffe_categories_map[key]);

        """
        # Testing:
        print ' '
        print 'imgList[0] ='
        print imgList[0]
        print 'key[0] = ' + imgList[0].split('.')[1][0:2]
        print 'labels[0] = ' + str(labels[0])
        sys.exit(0)
        """

    else:
        print 'Error - Unsupported dataset: ' + dataset;
        return None;
    
    # Make sure some dataset was imported:
    if len(imgList) <= 0:
        print 'Error - No images found in ' + str(dir)
        return None

    # Return list of filenames
    return imgList, labels



###############################################################################
# OpenCV face recognition and segmentation
###############################################################################


def DetectFace(image, faceCascade, gray):

    # Resize
    img = cv.resize(image, (0,0), fx=1, fy=1, interpolation = cv.INTER_CUBIC)
    
    # Convert to grayscale and equalize the histogram
    if gray:
        img = img.astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.equalizeHist(img, img)

    # Detect the faces
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.CASCADE_SCALE_IMAGE
    )
    
    print('Detected %d faces.' % len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return img, faces

# Crop image array to pixels indicated by crop box
def imgCrop(img, cropBox):
    x, y, w, h = cropBox
    img = img[y:(y+h), x:(x+h)]
    return img

# Convert bgr to rgb
def rgb(bgr_img):
    b,g,r = cv.split(bgr_img)       # get b,g,r
    rgb_img = cv.merge([r,g,b])     # switch it to rgb
    return rgb_img

# Given directory loc, get all images in directory and crop to just faces
# Returns face_list, an array of cropped image file names
def faceCrop(targetDir, imgList, faceCascade, color=True):

    # Iterate through images
    face_list = []
    for img in imgList:
        if os.path.isdir(img):
            continue
        pil_img = Image.open(img)
        if color:
            cv_img  = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        else:
            cv_img = np.array(pil_img)
        scaled_img, faces = DetectFace(cv_img, faceCascade, gray=False)
        
        # Iterate through faces
        n=1
        for face in faces:
            cropped_cv_img = imgCrop(scaled_img, face)
            if color:
                cropped_cv_img = rgb(cropped_cv_img)
            fname, ext = os.path.splitext(img)
            cropped_pil_img = Image.fromarray(cropped_cv_img)
            #save_name = loc + '/cropped/' + fname.split('/')[-1] + '_crop' + str(n) + ext
            save_name = targetDir + '/' + fname.split('/')[-1] + '_crop' + str(n) + ext
            cropped_pil_img.save(save_name)
            face_list.append(save_name)
            n += 1

    return face_list


# Delete all files in a directory matching pattern
def purge(dir, pattern):
    for f in os.listdir(dir):
    	if re.search(pattern, f):
    	    os.remove(os.path.join(dir, f))

# Delete a directory
def rmdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

# Create a directory. Overwrite any existing directories
def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)



###############################################################################
# Main
###############################################################################

### USER-SPECIFIED VARIABLES: ###

# List your dataset root directories here:
dirJaffe = 'jaffe'
dirCKPlus = None  # TODO
# dirOther ... TODO: allow any generic directory of pictures

# Select which dataset to use (case insensitive):
dataset = 'jaffe';

# Flags:
cropFlag = True # False disables image cropping

### START SCRIPT: ###

## Set up inputs
dir = None
if dataset.lower() == 'jaffe':
    dir = dirJaffe;
else:
    print 'Error - Unsupported dataset: ' + dataset
    sys.exit(0)
#dir = 'datasets/jaffe'
#dir = 'CKPlus/S010/006'

# Clean up and discard anything from the last run
dirCrop = dir + '/cropped';
rmdir(dirCrop);

# Load dataset image list
input_list, labels = importDataset(dir,dataset)

# Load Haar cascade files containing features
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)

# Perform detection and cropping if desired (and it should be desired)
if cropFlag:
    mkdir(dirCrop)
    input_list = faceCrop(dirCrop, input_list, faceCascade, color=False)


# Perform classification
classify_emotions(input_list, labels, show_confusion=False, show_faces=False, show_neurons=False)
