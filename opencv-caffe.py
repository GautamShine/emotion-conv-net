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

def showimage(img):
    if img.ndim == 3:
        img = img[:, :, ::-1]
    plt.set_cmap('jet')
    plt.imshow(img,vmin=0, vmax=0.2)
    
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

def mod_dim(img, x=256, y=256, c=3):
    resized = caffe.io.resize_image(img, (x,y,c)) # (256, 256, 3)
    rearranged = np.swapaxes(np.swapaxes(resized, 1, 2), 0, 1) # (3, 256, 256)
    return rearranged

def compute_mean(input_list):
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
        plt.imshow(np.swapaxes(np.swapaxes(mean, 0, 1), 1, 2))
        plt.show()
    return mean

def make_net(mean):
    caffe_root = '/home/gshine/Documents/caffe/'
    sys.path.insert(0, caffe_root + 'python')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

    net_root = '.'
    net_dir = 'VGG_S_rgb'

    net_pretrained = os.path.join(net_root, net_dir, 'EmotiW_VGG_S.caffemodel')
    net_model_file = os.path.join(net_root, net_dir, 'deploy.prototxt')
    VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
    return VGG_S_Net, categories

def plot_confusion_matrix(cm, names, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Add labels to confusion matrix:
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    
    plt.tight_layout()
    plt.ylabel('Correct label')
    plt.xlabel('Predicted label')
    plt.show()

def confusion(results, categories):
    map_categories = {
        'HA': categories.index('Happy'),
        'SA': categories.index('Sad'),
        'NE': categories.index('Neutral'),
        'AN': categories.index('Angry'),
        'FE': categories.index('Fear'),
        'DI': categories.index('Disgust'),
        'SU': categories.index('Surprise')
    }
    matrix = np.zeros((7,7))
    for label, pred in results:
        matrix[map_categories[label],pred] += 1
    
    # Display percent accuracy
    accuracy = float(np.trace(matrix))/len(results)
    print('Accuracy: ', accuracy*100, '%')
    print('Confusion Matrix: ')
    print(matrix)
    # Plot the confusion matrix
    plot_confusion_matrix(matrix, categories)

def classify_emotions(input_list, show_confusion, show_faces, show_neurons):
    # Compute mean
    mean = compute_mean(input_list)
    
    # Create VGG_S net with mean
    VGG_S_Net, categories = make_net(mean)

    # Classify images in directory
    conf_mat = [] # confusion matrix
    for img_file in input_list:
        input_image = caffe.io.load_image(img_file)
        # TODO: test larger crops + oversampling
        prediction = VGG_S_Net.predict([input_image], oversample=False)

        if show_confusion:
            label = img_file.split('.')[1][0:2]
            conf_mat.append((label, prediction.argmax()))

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

    if show_confusion:
        confusion(conf_mat, categories)

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

def imgCrop(img, cropBox):
    x, y, w, h = cropBox
    img = img[y:(y+h), x:(x+h)]
    return img

def rgb(bgr_img):
    b,g,r = cv.split(bgr_img)       # get b,g,r
    rgb_img = cv.merge([r,g,b])     # switch it to rgb
    return rgb_img

def faceCrop(loc, faceCascade, color=True):
    # Find all images in location
    imgList = glob.glob(loc+'/*')
    if len(imgList) <= 0:
        print 'No images found.'
        return

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
            save_name = loc + '/cropped/' + fname.split('/')[-1] + '_crop' + str(n) + ext
            cropped_pil_img.save(save_name)
            face_list.append(save_name)
            n += 1

    return face_list

###############################################################################
# Main
###############################################################################

def purge(dir, pattern):
    for f in os.listdir(dir):
    	if re.search(pattern, f):
    	    os.remove(os.path.join(dir, f))

def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

dir = 'datasets/jaffe'

# Clean up cropped images from directory
#purge('datasets/jaffe', 'crop')
mkdir(dir+'/cropped')

# Load Haar cascade files containing features
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)

# Perform detection and cropping
input_list = faceCrop(dir, faceCascade, color=False)

# Perform classification
classify_emotions(input_list, show_confusion=True, show_faces=False, show_neurons=False)
