"""
author: Alice Izsak
sudokureader.py asks for the file path for an image of a sudoku puzzle. If the path is valid, it will
display an image of the sudoku problem with the empty squares filled in correctly.
"""

import matplotlib.pyplot as plt
import cv2 as cv 
import numpy as np
import math
import tensorflow as tf
from sudoku import Sudoku

tf.enable_eager_execution()


def centerSudoku(image, preprocess = False):
    #Parameters: image, a string containing the file path of an image of a sudoku puzzle.
    #            preprocess, a boolean. If true the returned image is blurred and has thresholding applied to it.
    #Returns:    a numpy array, containing the pixel values of the image with the sudoku centered and squared.
    
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)  
    img = cv.GaussianBlur(img, (5, 5), 0)

    monochrome = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,9,2)

    proc = monochrome
    monochrome = cv.medianBlur(monochrome,5)
    monochrome = cv.bitwise_not(monochrome, monochrome)

    proc = cv.bitwise_not(proc, proc)
    kernel = np.ones((3,3),np.uint8)

    contours = cv.findContours(monochrome, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE )[0]


    ctr = max(contours, key=cv.contourArea)

    dr = max(ctr, key=lambda x:x[0][0]+x[0][1])
    dl = max(ctr, key=lambda x:x[0][0]-x[0][1])
    ur = max(ctr, key=lambda x:-x[0][0]+x[0][1])
    ul = max(ctr, key=lambda x:-x[0][0]-x[0][1])

    pts1 = np.float32([dr[0],dl[0], ur[0], ul[0]])
    pts2 = np.float32([[300,300],[300,10],[10,300],[10,10]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    
    if preprocess:
        monochrome = cv.cvtColor(monochrome, cv.COLOR_GRAY2BGR)
        dst = cv.warpPerspective(monochrome,M,(310,310))
        monochrome = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        return monochrome
    else:
        img = cv.imread(image, cv.IMREAD_COLOR)
        dst = cv.warpPerspective(img,M,(310,310))
        return dst

def digitFinder(image, showDigits = False):
    #Parameters: image, a string, the file path for a sudoku image file.
    #            showDigits, a boolean indicating whether or not to draw the digits
    #Returns: a dict where the keys are of the form (i,j) and the values are numpy array versions of the image
    #contained in cell i,j of the sudoku puzzle.
    
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)  
    if img is None:
        print("No image found at that file path.")
        return None

    monochrome = centerSudoku(image, True)
    contours = cv.findContours(monochrome, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[0]
    rangeCtrs = range(len(contours))
    largeCtrs = [i for i in rangeCtrs if cv.contourArea(contours[i]) > 5 and cv.contourArea(contours[i]) <800]
    
    #Next look for contours whose bounding rectangle is contained within a cell of the sudoku puzzle
    boundCtrs = {i: cv.boundingRect(contours[i]) for i in largeCtrs}
    cellCenters = {(i,j): (10 + 290/18 + i*290/9 , 10 + 290/18 + j*290/9) for i in range(9) for j in range(9)}
    
    #We pad the width of the cell with a margin so as to not include sudoku grid lines
    margin = 12
    cellULCorner = {(i,j): (10 + margin + i*290/9 , 10 + margin + j*290/9) for i in range(9) for j in range(9)}
    cellLength = 290/9 - margin
    digits = {(i,j) : [] for i in range(9) for j in range(9)}
    monochrome = cv.cvtColor(monochrome, cv.COLOR_GRAY2BGR)
    
    for i in range(9):
        for j in range(9):
            inCell = []
            for ctr in largeCtrs:
                x, y, w, h = boundCtrs[ctr]
                ULBound = 30 + x > cellULCorner[(i,j)][0] and 30 + y > cellULCorner[(i,j)][1]
                DRBound = x + w < 30 + cellULCorner[(i,j)][0] + cellLength and y + h < 30 + cellULCorner[(i,j)][1] + cellLength
                if ULBound and DRBound:
                    ULBound = x < cellCenters[(i,j)][0] and y < cellCenters[(i,j)][1]
                    DRBound = x + w > cellCenters[(i,j)][0] and y + h > cellCenters[(i,j)][1]
                    inCell = inCell + [ctr]
                    
            #Also include contours close to the center of the cell. Some digits may be too warped to fit in the 
            #bounding box approach we did earlier, this step should include them.
            for cont in inCell:
                center = np.array(cellCenters[(i,j)])
                dist = [np.linalg.norm(point - center) for point in contours[cont]]
                if any([d  < cellLength/3 for d in dist]):
                    digits[(i,j)] = digits[(i,j)] +[cont]
                    
    digictrs = [contours[i] for sublist in digits.values() for i in sublist]

    def encloseRect(rects):   
    #Parameters: rects, a list of bounding boxes of every contour within a cell
    #Returns: [x, y, rightx, bottomy], a list of 4 ints that describe the box that bounds all contours in the cell.
    #x,y is the upper left coordinate of the box and rightx, bottomy are bottom right.
    
        xs = [rect[0] for rect in rects]
        ys = [rect[1] for rect in rects]
        rightxs = [rect[0] + rect[2] for rect in rects]
        bottomys = [rect[1] + rect[3] for rect in rects]
        x = min(xs)
        y = min(ys)
        rightx = max(rightxs)
        bottomy = max(bottomys)
        return [x, y, rightx, bottomy]
    
    digitsBound = {}
    monochrome = cv.cvtColor(monochrome, cv.COLOR_BGR2GRAY)

    for i in digits:
        if len(digits[i]) > 0:
            rects = [boundCtrs[ctr] for ctr in digits[i]]
            digitsBound[i] = encloseRect(rects)
            
    images = {}
    #Scale the digit images we found to all be 28 by 28 pixels
    for i in digits:
        if len(digits[i]) == 0:
            images[i] = np.zeros((28, 28), np.uint8)
        else:
            x, y, rightx, bottomy = digitsBound[i]
            crop = monochrome[y:bottomy+1, x:rightx+1]
            h = bottomy - y
            w = rightx - x
            scaleFactor = 22.0 / h
            scaledW = int(w * scaleFactor)
            oddWidth = scaledW % 2
            xmargin =(28 - scaledW) / 2
            if xmargin > 0:
                resized = cv.resize(crop,(scaledW, 22), interpolation = cv.INTER_LINEAR)
                digit = cv.copyMakeBorder(resized,3,3,xmargin,xmargin + oddWidth,cv.BORDER_CONSTANT,value=0)
                images[i] = digit
            else:
                images[i] = np.zeros((28, 28), np.uint8)
                
    if showDigits:
        for i in range(9):
            for j in range(9):   
                plt.subplot(9,9,1+j+9*i)
                plt.imshow(images[(j,i)], cmap = 'gray') 
                plt.xticks([]), plt.yticks([])
        plt.show()
    imagesPrime = {(j, i): images[(i,j)] for i in range(9) for j in range(9)}
    return imagesPrime

    

def sudokuImageSolver(image):
    #Parameters: image, a string, the file path for a sudoku image file.
    #Given an image of an unsolved Sudoku puzzle, applies our convnet model to read the puzzle, solves it,
    #and draw the solved puzzle
    
    digits = digitFinder(image)
    for i in range(9):
        for j in range(9):
            if i == 0 and j == 0:
                digitsArray = digits[(i,j)]
                digitsArray.astype(float)
                digitsArray =  np.expand_dims(digitsArray, axis=0)
                digitsArray =  np.expand_dims(digitsArray, axis=3)
                digitsArray = np.divide(digitsArray, 255.0)
            else:
                digit = digits[(i,j)]
                digit.astype(float)
                digit = np.divide(digit, 255.0)
                digit = np.expand_dims(digit, axis=0)
                digit = np.expand_dims(digit, axis=3)
                digitsArray = np.concatenate((digitsArray, digit), axis=0)
    digitsDS = tf.data.Dataset.from_tensor_slices(digitsArray)
    model = tf.keras.models.load_model("/Users/aizsak/Sudoku Dataset/ckpt")
    digitsDS = digitsDS.batch(81)
    predictions = model.predict(digitsDS)
    labels = np.zeros(81)
    for i in range(81):
        labels[i] = np.argmax(predictions[i])
    labels = labels.reshape((9, 9))
    dictPuzzle ={(i,j):(set(range(1,10)) if labels[i,j]==0 else {int(labels[i,j])}) for i in range(9) for j in range(9)}
    sudo = Sudoku()
    sudo.fromDict(dictPuzzle)
    solved = sudo.solver()
    sudo.fromDict(solved)
    print(sudo)
    centered = centerSudoku(image)
    drawSolution(centered, labels, solved)

def drawSolution(img, labels, solution):
    #Given a centered image of an unsolved Sudoku puzzle with labeled cells and solution, creates and displays an image
    #of the image with all cells filled in.
    font = cv.FONT_HERSHEY_DUPLEX
    scale = 0.7
    color = (255,0,255)
    cellSize = 290.0 / 9
    line = 2
    for i in range(9):
        for j in range(9):
            if labels[i,j]==0:
                bottomLeftCornerOfText = (10 + int(j*cellSize) + 6 , 10 + int((i + 1)*cellSize) - 8)
                text = str(list(solution[(i,j)])[0])
                cv.putText(img, text, bottomLeftCornerOfText, font, scale, color, line)
    plt.axis("off")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) 
    plt.show()



if __name__ == "__main__":
    image = raw_input("Enter the file path for an image of a sudoku puzzle: ")
    if cv.imread(image, cv.IMREAD_GRAYSCALE) is None:
        print("Sorry, I can't read that file.")
    else:
        sudokuImageSolver(image)


