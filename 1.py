# System utilities
import os
import glob

# Data manipulation and visualization libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Scikit-learn (machine learning utilities)
from sklearn.model_selection import train_test_split

# OpenCV for image processing
import cv2 as cv

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_action = pd.read_csv("/Users/akhildnair/Downloads/human-action-recognition-main/data/Testing_set.csv")
train_action = pd.read_csv("/Users/akhildnair/Downloads/human-action-recognition-main/data/Training_set.csv")

print(train_action.head())

img = cv.imread('data/train/' + train_action.filename[0])
print(plt.title(train_action.label[0]))
print(plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)));
print(test_action.shape)

def show_img_train():
    img_num = np.random.randint(0, len(train_action))
    img_path = 'data/train/' + train_action.filename[1000]
    img = cv.imread("/Users/akhildnair/Downloads/human-action-recognition-main/resized-test/Image_1000.jpg")
    
    # Display the image with title
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(train_action.label[1000])
    plt.axis('off')  # Hide axes for better visualization
    plt.show()  # Required to render the plot

def show_img_test():
    img_num = np.random.randint(0, len(test_action))
    img_path = 'data/test/' + test_action.filename[1000]
    img = cv.imread("/Users/akhildnair/Downloads/human-action-recognition-main/resized-test/Image_1000.jpg")

    # Display the image
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(test_action.label[1000])
    plt.axis('off')  # Hide axes for better visualization
    plt.show()  # Required to render the plot