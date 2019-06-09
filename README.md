# ECE285 Final Project 
## Description
This is project Style Transfer developed by team Moha composed of Changyu Li, Jiayu He and Wenda Chen.

## Requirement
None

## Code organization
<pre>
285FinalProject_part1.ipynb ------ Run a demo of our code for part1  
285FinalProject_part1_exp.ipynb -- Run the training of our task1 (reproduces Figure 3 and 4 of our report)  
285FinalProject_part1.py --------- Python file for the part 1 demo code  
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
                              
