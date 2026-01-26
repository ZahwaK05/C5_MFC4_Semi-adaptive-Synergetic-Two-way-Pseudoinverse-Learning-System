## Team Members:

Annem Sai Reddy  
CB.SC.U4AIE24205  

Devana Madhavan Nambiar  
CB.SC.U4AIE24213  

Tharappel Manas  
CB.SC.U4AIE24257  

Zahwa K  
CB.SC.U4AIE24261  

## Semi-adaptive Synergetic Two-way Pseudoinverse Learning System

  Base paper : https://arxiv.org/pdf/2406.18931 
## Project Outline : 

This project implements a Semi-adaptive Synergetic Two-way Pseudoinverse Learning System (SLS-TPLS) as a non-gradient deep learning framework. The primary goal of the project is to overcome the limitations of traditional gradient-descent–based deep learning methods, such as high training time, difficulty in hyperparameter tuning, and fixed network architecture.

The system is built using multiple synergetic subsystems, where each subsystem contains:

Forward learning module using stacked Pseudoinverse Learning Autoencoders (PILAE)

Backward learning module that propagates label information backward through the network

Feature fusion module that concatenates forward and backward learned features

Each subsystem operates independently on sampled data and is trained using non-gradient optimization techniques such as pseudoinverse computation and FISTA. The architecture of the network is semi-adaptive, meaning the depth of the model is determined automatically using an early stopping strategy instead of manual tuning.

The outputs from all subsystems are combined to form a robust and efficient learning system that achieves high classification accuracy while significantly reducing training time.
## Project Update 


Implemented a semi-adaptive synergetic two-way pseudoinverse learning model

Used forward and backward learning without gradient descent

Applied the model on the MNIST dataset

Evaluated performance based on classification accuracy and efficiency




## Challenges / Issues Faced

Understanding the theoretical concepts of non-gradient pseudoinverse learning was initially challenging

Implementing forward and backward learning without backpropagation required careful handling of matrix dimensions

Managing numerical stability and memory usage during pseudoinverse computations

Debugging inconsistent results across different datasets and parameter settings



## Future plans

Future Work

Implement new methods in the feature fusion module

Increase the number of neurons to improve learning capacity

Apply the model to more datasets

Test the system using different hyperparameter settings

## folder structure

<pre>
C5_MFC4_Semi-adaptive-Synergetic-Two-way-Pseudoinverse-Learning-System/
├── CODE/
│   ├── Dataset/
│   │   ├── mnistX.mat
│   │   └── mnistnumY.mat
│   ├── ActivationFunc.m
│   ├── DeactivationFunc.m
│   ├── calculateWeights4AE.m
│   ├── finetunning.m
│   ├── fusionnet.m
│   ├── initInputWeight.m
│   ├── PILAE.m
│   ├── S2WPILS_demo_MNIST.m
│   ├── targetPrepro.m
│   ├── train_SHLNN.m
│   └── test_SHLNN.m
├── README.md
└── .git/
└── PPT

</pre>


