###############################################################################
# Utility functions for OpenCV-Caffe chaining
###############################################################################

import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Image
import caffe

# Plot confusion matrix
def plot_confusion_matrix(cm, names=None, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(4)
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

# Generate confusion matrix for Jaffe
# results = list of tuples of (correct label, predicted label)
#           e.g. [ ('HA', 3) ]
# categories = list of category names
# Returns confusion matrix; rows are correct labels and columns are predictions
def confusion_matrix(results, categories, plotConfusion=False):
    # Empty confusion matrix
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

# Get entire dataset
# Inputs: Dataset root directory; optional dataset name
# Returns: List of all image file paths; list of correct labels for each image
def importDataset(dir, dataset, categories):
    imgList = glob.glob(dir+'/*')
    labels = None

    # Datset-specific import rules:
    if dataset.lower() == 'jaffe':
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
            if os.path.isdir(img):
                continue
            key = img.split('.')[1][0:2]
            labels.append(jaffe_categories_map[key]);
    elif dataset.lower() == 'misc':
        labels = [0,1,2,3,4,5,6]
    else:
        print 'Error - Unsupported dataset: ' + dataset;
        return None;

    # Make sure some dataset was imported
    if len(imgList) <= 0:
        print 'Error - No images found in ' + str(dir)
        return None

    # Return list of filenames
    return imgList, labels

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

def flatten(biglist):
    return [item for sublist in biglist for item in sublist]
