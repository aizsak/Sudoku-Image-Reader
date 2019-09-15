# Sudoku Image Reader

#### Overview 

sudokureader.py takes an image of a sudoku puzzle identifies the puzzle with tools from opencv, uses a convnet to read the digits and then 
uses constraint propagation to solve it. Here are the steps the code takes:
1. Preprocesses the image by blurring and performing adaptive thresholding
2. Finds the largest contour in the image. We assume this is our sudoku puzzle grid. 
3. Geometrically transforms the image so that large contour is squared and centered.
4. Splits the image into 9 by 9 cells. 
5. Look for contours in each cell. I don't include any contours so small they are likely noise. I also avoid contours close to
outer portion of cell since those are likely parts of the sudoku grid. Now we have our digit images.
6. Resize and center the digit images.
7. Use a convolutional neural network to identify the digits.
8. Perform constraint propagation on the sudoku puzzle.
9. If the puzzle is not solved, we do a depth first search on the solution space until we find a solution.

### Requires
Tensorflow, OpenCV, matplotlib, numpy 

### To Run

Type in terminal `python sudokureader.py`

### Also Included

A sudoku dataset folder, which has labeled digit images taken from images of sudoku puzzles. prepdataset.py  
created the digit dataset, preprocessed the images, and split the data into testing, training and validation data sets.

sudocnn.py trains the convnet model. It also gives the loss and accurary of our model on our testing, training and 
validation datasets.

The file ckpt in the sudoku dataset folder is a copy of our trained convnet model. It has  accuracy %99.65 on the test set.


#### Example Input

![input](https://github.com/aizsak/Sudoku-Image-Reader/blob/master/sampleinput.jpg)

#### Example Output

![output](https://github.com/aizsak/Sudoku-Image-Reader/blob/master/sampleoutput.png)

### Acknowledgment
The sudoku dataset is composed of sudoku images created and labeled by Baptiste Wicht, you can find their dataset at 
[this link](https://github.com/wichtounet/sudoku_dataset). Project is inspired by OpenCV's guide on geometric transformations,
which includes an example of a sudoku puzzle being centered and squared.



