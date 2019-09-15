
"""
author: Alice Izsak
sudocnn.py trains our model using the dataset of labelled digits from sudoku puzzles. It also gives the loss and 
accurary of our model on our testing, training and validation datasets.

If you use this code on your own system, you will need to change the variable parent to be the path for your 
Sudoku dataset.
"""

import tensorflow as tf
import random 
import glob
import os
import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2 as cv
import numpy as np

tf.enable_eager_execution()

parent = "/Users/aizsak/"

def preprocessImage(imagePath):
    image = tf.image.decode_jpeg(imagePath, dct_method = "INTEGER_ACCURATE")
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize 
    return image

def loadAndPreprocess(path):
    image = tf.read_file(path)
    return preprocessImage(image)

def trainModel():
    #Trains model using our training set. Saves a copy of model as ckpt
    imagePaths = glob.glob("Sudoku Dataset/train/*/*.jpg")
    imagePaths = [parent + path for path in imagePaths]
    random.shuffle(imagePaths)
    allLabels = [int(os.path.dirname(path)[-1]) for path in imagePaths]

    pathDS = tf.data.Dataset.from_tensor_slices(imagePaths)

    imageDS = pathDS.map(loadAndPreprocess)
    
    def displayDSImages(imageDS, allLabels):
        #Displays first 4 images with labels from image dataset. Only used for testing.
        
        plt.figure(figsize=(2,2))
        for n,img in enumerate(imageDS.take(4)):
            img = tf.squeeze(img, [2])
           
            plt.subplot(2,2,n+1)
            plt.imshow(img, cmap='gray')
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(allLabels[n])
        plt.show()
        
    classDS = tf.data.Dataset.from_tensor_slices(tf.cast(allLabels, tf.int64))
    trainDS = tf.data.Dataset.zip((imageDS, classDS))
    DS_SIZE = 5670
    BATCH_SIZE = 1000
    trainDS = trainDS.shuffle(buffer_size=DSsize)
    trainDS = trainDS.repeat()
    trainDS = trainDS.batch(1000)
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    trainDS = trainDS.prefetch(buffer_size=BATCH_SIZE)
    features, labels = next(iter(trainDS))
    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(features, labels, epochs=5)

    model.save(parent + "Sudoku Dataset/ckpt")

def testModel():
    #Prints the accuracy and loss of the model on the test dataset
    testPaths = glob.glob("Sudoku Dataset/test/*/*.jpg")
    testPaths = [parent + path for path in testPaths]
    allLabels = [int(os.path.dirname(path)[-1]) for path in testPaths]

    pathDS = tf.data.Dataset.from_tensor_slices(testPaths)

    imageDS = pathDS.map(loadAndPreprocess)
    classDS = tf.data.Dataset.from_tensor_slices(tf.cast(allLabels, tf.int64))

    testDS = tf.data.Dataset.zip((imageDS, classDS))
    DSsize = 5670

    model = tf.keras.models.load_model(parent + "Sudoku Dataset/ckpt")
    BATCH_SIZE = len(testPaths)
    test_accuracy = tf.contrib.eager.metrics.Accuracy()
    testDS = testDS.batch(BATCH_SIZE)
    testFeatures, testLabels = next(iter(testDS))
    test_loss, test_acc = model.evaluate(testFeatures, testLabels)
    print("Accuracy of model on testing set is " + str(test_acc))
    print("Loss of model on testing set is " + str(test_loss))
    
def valModel():
    #Prints the accuracy of the model on the test dataset
    testPaths = glob.glob("Sudoku Dataset/val/*/*.jpg")
    testPaths = [parent + path for path in testPaths]

    allLabels = [int(os.path.dirname(path)[-1]) for path in testPaths]

    pathDS = tf.data.Dataset.from_tensor_slices(testPaths)

    imageDS = pathDS.map(loadAndPreprocess)
    classDS = tf.data.Dataset.from_tensor_slices(tf.cast(allLabels, tf.int64))

    testDS = tf.data.Dataset.zip((imageDS, classDS))
    DSsize = 5670

    model = tf.keras.models.load_model(parent + "Sudoku Dataset/ckpt")
    BATCH_SIZE = len(testPaths)
    test_accuracy = tf.contrib.eager.metrics.Accuracy()
    testDS = testDS.batch(BATCH_SIZE)
    testFeatures, testLabels = next(iter(testDS))
    test_loss, test_acc = model.evaluate(testFeatures, testLabels)
    print("Accuracy of model on validation set is " + str(test_acc))



if __name__ == "__main__":
    trainModel()
    testModel()
    valModel()

