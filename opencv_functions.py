###############################################################################
# OpenCV face recognition and segmentation
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

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
