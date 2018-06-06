# Machine-Operating-Status-Classification_Sound-Recognition
## 1. Goal and Data Description

The practice is described as follows:

In order to operate a group of equipment more efficiently, it is crucial to identify their operation status. The goal is to identify at each time point what are the machines that
are functioning by their sound.

Individual equipment operate status in each recording were annotated by the same person. The annotator was instructed to annotate
all audible equipment operate status, decide the start time and end time of the operate status as it fit, and choose following event labels:

* *equipment 1*
* *equipment 2*
* *equipment 3*
* *equipment 4*

**Training Data**

The training dataset is an audio file formatted "MPEG-4 Audio", with time duration 12 minutes and 4 seconds. It is a supervised dataset and the audio file is 
marked by operation status during specific time span. The picture below is a snapshot of the training dataset:

![1](https://user-images.githubusercontent.com/38633055/41045726-80f596a2-6998-11e8-94f3-f3939683d4a7.PNG)

## 2. Executive Summary

Without much experience in dealing with audio dataset, I started this project by figuring out how sound is stored in an audio file and how to transform sound into numerical data that I am 
familiar with. Briefly speaking, the outline of this project includes:

* Read the audio file to obtain initial signal and sample rate
* Split continuous time into discrete sample bins
* Build features by MFCC decomposition
  * Approach 1: set window step as 0.0001 for Neural Nets
  * Approach 2: set window step as 0.01 for simpler models
* Define response variable
* Train test split to measure model performance
* Apply Random Forest model
* Apply 1D Convolutional Neural Network
* Predict first 20 mins of test set to submit 

## 3. Read the audio file

The training set is an audio file named "YT.m4a" and python has many handy packages to process files in "wav" format, so the first thing I do is to
transform the format of this file into "wav" format.

![2](https://user-images.githubusercontent.com/38633055/41050706-b0035320-69a3-11e8-89aa-79f2e21a42c4.PNG)

But I noticed the file size has increased from 5M to 68M.









 


