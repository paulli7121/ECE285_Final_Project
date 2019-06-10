# ECE285 Final Project 
## Description
This is project Style Transfer developed by team Moha composed of Changyu Li, Jiayu He and Wenda Chen. 

## Requirement
Pytorch, please make sure the training data set (COCO_2015) is at /datasets/COCO-2015 directory. To 
run the demo, please download all the files(especially the file ResNetTrained, it contains the trained parameters) ,makesure the images are in the same directory of notebook file and run the demo files.

## Code organization
<pre>
style------------------------------Folder contain the training images
Task1_Demo.ipynb ----------------- Run a demo of our code for part1  
Task1_training.ipynb ------------- Run the training of our task1 (reproduces Figure 3 and 4 of our report)  
Task1.py ------------------------- Python file for the part 1 demo code
house.jpg--------------------------content image for demo 1
starry_night.jpg------------------ style image for demo 1
ResNetTrained -------------------- The file stored the trained resnet parameters, create an instance of type 
                                   ResNetGen and load this file to get the trained network
ResBlock.py ---------------------- The python file which contained the design of a residual block used in 
                                   the project
ResNetGenerator.py --------------- The python file which contained the design of a ResNet, which is used as 
                                   transform function 
Task2_Training.ipynb ------------- The training program of task 2
Task2_Demo.ipynb ----------------- The Demo program of task 2. The program loads the trained learning parameters 
                                   stored by the file ResNetTrained. 
</pre>
                              
