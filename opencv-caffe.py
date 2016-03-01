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

import os, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

###############################################################################
# Caffe VGG_S net emotion classification
###############################################################################

caffe_root = '/home/gshine/Documents/caffe/'
sys.path.insert(0, caffe_root + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

net_root = '.'

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

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

cur_net_dir = 'VGG_S_rgb'

mean_filename=os.path.join(net_root,cur_net_dir,'mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

net_pretrained = os.path.join(net_root,cur_net_dir,'EmotiW_VGG_S.caffemodel')
net_model_file = os.path.join(net_root,cur_net_dir,'deploy.prototxt')
VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

def classify_emotions(input_list, show_faces, show_neurons):
    for image_file in input_list:
        input_image = caffe.io.load_image(image_file)
        # TODO: test larger crops + oversampling
        prediction = VGG_S_Net.predict([input_image], oversample=False)
        print 'prediction: {0}'.format(categories[prediction.argmax()])
        
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

def faceCrop(loc, faceCascade):
    # Find all images in location
    imgList = glob.glob(loc)
    if len(imgList) <= 0:
        print 'No images found.'
        return
    
    # Iterate through images
    face_list = []
    for img in imgList:
        pil_img = Image.open(img)
        cv_img  = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
        scaled_img, faces = DetectFace(cv_img, faceCascade, gray=False)
        
        # Iterate through faces
        n=1
        for face in faces:
            cropped_cv_img = imgCrop(scaled_img, face)
            cropped_cv_img = rgb(cropped_cv_img)
            fname, ext = os.path.splitext(img)
            cropped_pil_img = Image.fromarray(cropped_cv_img)
            save_name = fname + '_crop' + str(n) + ext
            cropped_pil_img.save(save_name)
            face_list.append(save_name)
            n += 1

    return face_list

def purge(dir, pattern):
    for f in os.listdir(dir):
    	if re.search(pattern, f):
    	    os.remove(os.path.join(dir, f))

# Clean up cropped images from directory
purge('images/', 'crop')

# Load Haar cascade files containing features
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv.CascadeClassifier(cascPath)

# Perform detection and cropping
input_list = faceCrop('images/*', faceCascade)

# Perform classification
classify_emotions(input_list, show_faces=False, show_neurons=False)
