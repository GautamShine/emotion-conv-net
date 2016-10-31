import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

from caffe_functions import *
from opencv_functions import *
from utility_functions import *

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

plotSideBySide = True # Plot before/after images together?
saveDir = 'test_screenshots' # Folder to save screenshots to

useCNN = True # Set to false to simply display the default emoji
defaultEmoji = 2 # Index of default emoji (0-6)

### START SCRIPT ###

# Set up face detection
faceCascades = load_cascades()

# Set up network
if useCNN:
  mean = loadMeanCaffeImage()
  VGG_S_Net = make_net(mean,net_dir="Custom_Model")

# Get all emojis
emojis = loadAllEmojis()

# Set up display window
cv.namedWindow("preview")

# Open input video steam
vc = cv.VideoCapture(0)

# Check that video stream is running
if vc.isOpened(): # try to get the first frame
  rval, frame = vc.read()
  #frame = frame.astype(np.float32)
else:
  rval = False

while rval:
  # Mirror image
  frame = np.fliplr(frame)
  
  # Detect faces
  detect = True
  if detect:
    # Find all faces
    with nostdout():
      _, faces = DetectFace(frame,True,faceCascades,single_face=False,second_pass=False,draw_rects=False,scale=1.0)
    #frame = addEmoji(frame,faces,emoji)

    oldFrame = frame.copy()
    if len(faces) == 0 or faces is None:
      # No faces found
      pass
    else:
      # Toggle whether to do dynamic classification, or just to display one user-picked emoji
      useCNN = False

      if useCNN:
        # Get a label for each face
        labels = classify_video_frame(frame, faces, VGG_S_Net, categories=None)

        # Add an emoji for each label
        frame = addMultipleEmojis(frame,faces,emojis,labels)
        
        # Print first emotion detected
        #print categories[labels[0]]

      else:
        # Just use the smiley face (no CNN classification)
        frame = addEmoji(frame,faces,emojis[defaultEmoji])

  # Show video with faces
  if plotSideBySide:
    img = cvCombineTwoImages(oldFrame,frame)
    cv.imshow("preview", img)
  else:
    img = frame.copy()
    cv.imshow("preview", img)

  # Read in next frame
  rval, frame = vc.read()

  # Wait for user to press key. On ESC, close program
  key = cv.waitKey(20)
  if key == 27: # exit on ESC
    break
  elif key == 115 or key == 83: # ASCII codes for s and S
    filename = saveTestImage(img,outDir=saveDir)
    print "Image saved to ./" + filename

cv.destroyWindow("preview")
