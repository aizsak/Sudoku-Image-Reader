
"""
This file prepares our dataset. It uses a dataset of labeled sudoku puzzles to extract a dataset of labeled digits
from the sudoku puzzles. It then splits our dataset into training, testing and validation datasets.
"""

import glob
import os
import random
from shutil import copyfile
from sudokuread import digitFinder

images = glob.glob("Sudoku Dataset/*.jpg")


def createFolders():
    #Create folders named '0' through '9' in our dataset folder. Folder name will be the label  for images in folder.
    parent = "Sudoku Dataset/"
    for i in range(10):
        os.mkdir(parent + "train/" + str(i))
        
        
def labelDigits(imgName):
    #Given a labelled sudoku image, save labelled images of each of the digits. 
    digitImgs = digitFinder(imgName)
    dataName = imgName[:-4] + ".dat"
    f = open(dataName, "r")
    text = f.read().split("\n")
    for i in range(9):
        line = text[2+i]
        for j in range(9):
            digitNum = line[2*j]
            destination = r"Sudoku Dataset/" + str(digitNum) + "/"
            
            fileName = len(glob.glob(destination + r"*.jpg"))
            destination = destination +imgName[-8:-4]+"X" +str(fileName) +r".jpg"
            cv.imwrite(destination, digitImgs[(i,j)])      
    f.close()
    

def splitDS():
    #Split the dataset into training, testing and validation datasets
    paths = glob.glob("Sudoku Dataset/*/*.jpg")
    random.shuffle(paths)
    numPaths = len(paths)
    numTrain = int(0.6*numPaths)
    numTest = int(0.2*numPaths)
    numVal = numTest
    parent = "Sudoku Dataset/"
    for path in paths[:numTrain]:
        label = os.path.dirname(path)[-1]
        fileName = os.path.basename(path)
        dst = parent + "train/" +label+ "/" + fileName
        copyfile(path, dst)
    for path in paths[numTrain:numTrain + numTest]:
        label = os.path.dirname(path)[-1]
        fileName = os.path.basename(path)
        dst = parent + "test/" +label+ "/" + fileName
        copyfile(path, dst)
    for path in paths[numTrain + numTest:]:
        label = os.path.dirname(path)[-1]
        fileName = os.path.basename(path)
        dst = parent + "val/" +label+ "/" + fileName
        copyfile(path, dst)
        
        
        
if __name__ == "__main__":
    createFolders()
    for image in images:
        labelDigits(image)
    splitDS()

