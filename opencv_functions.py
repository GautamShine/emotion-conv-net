###############################################################################
# OpenCV face recognition and segmentation
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

from utility_functions import *

def load_cascades():
    # Load Haar cascade files containing features
    cascPaths = ['haarcascades/haarcascade_frontalface_default.xml',
                 'haarcascades/haarcascade_frontalface_alt.xml',
                 'haarcascades/haarcascade_frontalface_alt2.xml',
                 'haarcascades/haarcascade_frontalface_alt_tree.xml'
                 'lbpcascades/lbpcascade_frontalface.xml']
    faceCascades = []
    for casc in cascPaths:
        faceCascades.append(cv.CascadeClassifier(casc))

    return faceCascades

def DetectFace(image, color, faceCascades, second_pass, draw_rects):
    # Resize
    img = cv.resize(image, (0,0), fx=1, fy=1, interpolation = cv.INTER_CUBIC)

    # Convert to grayscale and equalize the histogram
    if color:
        gray_img = img.copy().astype(np.uint8)
        gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
    cv.equalizeHist(gray_img, gray_img)


    # Eliminate spurious extra faces
    discardExtraFaces = False   # Set to true to enable
    if discardExtraFaces and faces.shape[0] > 1:
        faces = faces[0,:]
        faces = faces[np.newaxis,:]



    # Detect the faces
    faces = faceCascades[2].detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(50, 50),
        flags = cv.CASCADE_SCALE_IMAGE)

    print('Detected %d faces.' % len(faces))
    # Draw a rectangle around the faces
    if draw_rects:
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if len(faces) > 0 and second_pass:
        approved = []
        for i in range(len(faces)):
            cropped_face = imgCrop(gray_img, faces[i])
            alt_check = faceCascades[1].detectMultiScale(
                cropped_face,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(int(0.8*faces[i][2]), int(0.8*faces[i][3])),
                flags = cv.CASCADE_SCALE_IMAGE)
            # Check if exactly 1 face was detected in cropped image
            if len(alt_check) == 1:
                approved.append(i)
        faces = faces[approved]
        
    return img, faces

# Crop image array to pixels indicated by crop box
def imgCrop(img, cropBox, scale=1.0):
    x, y, w, h = cropBox
    if scale != 1.0:
        x += int(w*(1-scale)/2)
        y += int(h*(1-scale)/2)
        w = int(w*scale)
        h = int(h*scale)
    img = img[y:(y+h), x:(x+h)]
    return img

# Convert bgr to rgb
def rgb(bgr_img):
    b,g,r = cv.split(bgr_img)       # get b,g,r
    rgb_img = cv.merge([r,g,b])     # switch it to rgb
    return rgb_img

# Given directory loc, get all images in directory and crop to just faces
# Returns face_list, an array of cropped image file names
def faceCrop(targetDir, imgList, color=True):
    # Load list of Haar cascades for faces
    faceCascades = load_cascades()

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
        scaled_img, faces = DetectFace(cv_img, color, faceCascades, second_pass=False, draw_rects=False)

        # Iterate through faces
        n=1
        for face in faces:
            cropped_cv_img = imgCrop(scaled_img, face, scale=1.0)
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
