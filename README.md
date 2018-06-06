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

## 3. Read The Audio File

The training set is an audio file named "YT.m4a" and python has many handy packages to process files in "wav" format, so the first thing I do is to transform the format of this file into "wav" format.

![2](https://user-images.githubusercontent.com/38633055/41050706-b0035320-69a3-11e8-89aa-79f2e21a42c4.PNG)

I noticed the file size had increased from 5M to 68M due to the format change.

To read the ".wav" file in python I loaded libraries like *scipy* and *python_speech_features*

![3](https://user-images.githubusercontent.com/38633055/41056422-a6729626-69b3-11e8-8138-07c57a674008.PNG)

after reading the file I obtain two variables. Rate reflects the sample rate of the audio, which is 44100Hz in this case. Another variable gives the signal of data, which is a N-by-2 matrix. the number N is actually the sample rate times number of seconds. The figure below shows how I verify the length of audio by these two variables

![4](https://user-images.githubusercontent.com/38633055/41057007-6dd3c82e-69b5-11e8-907e-3add603be653.PNG)

## 4. Split Continuous Time Into Sample Bins

Because the original audio recorded is continous in time and in order to analyze machine operating status at any given time(period) I set the sample bin to be 0.01 seconds. For this 12-minute audio I therefore have 72426 sample bins. In the test set, I will also predict operating equipment in each 0.01s. 

## 5. Build Features By MFCC Decomposition

In this section, I will build features for our classification problem. As explained before, using package *Scipy.io.wavefile.read* will return two variables- sample rate and data read from wav file which is a N-by-2 matrix. The signal is later translated to 13 features of sound by MFCC approach. In sound processing , the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. MFCC are coefficients that collectively make up an MFC. MFCC are commonly used as features in speech recognition systems. 

I load python library *python_speech_features* to apply MFCC feature extraction. The key parameters to use this method are *"samplerate*" and "*winstep*". I have two samples of MFCC application to our signal data for different models. Because the sample rate I applied for reading wav file is 44100, I also use this value for MFCC. At this moment you might already notice that with the same sample rate, if I set winstep as 0.01s(sample bin) I would obtain the same row size N for the final MFCC output. Similarly if parameter "winstep" is 0.0001 I would obtain a 100-by-13 matrix for each sample bin, thus by setting "winstep" as 0.0001 the final dataset will contain 72426 sample bins and each bin will be a 100-by-13 matrix. I used first sample for random forest classification and second sample for nueral nets. I will explain more in the modeling section.  

![5](https://user-images.githubusercontent.com/38633055/41060003-c61603a0-69bd-11e8-8ebb-88b4b6459bf6.PNG)

From Figure above, *sig_train* is a three-dimension matrix containing original signal and *mfcc_train* is also a three-dimention matrix denoting MFCC features.

## 6. Define Response Variable

I look at this problem as a multi-label classification question. I have thought about two options to define the target variable. The first option came to me was that I use 16 labels, because there are 4 pieces of equipment and there could be 0, 1, 2, 3 or 4 pieces of machines working at the same time. By defining it in this way, it will turns to be a one-label multiclass classification problem. I didn't select this method because there are only 7 status in the training set and 9 other status do not have data to be trained.

Thus I define the target variable y as a 4-dimension vector. each element is either 0 or 1. For example if all the four machines are functioning, status would be [1,1,1,1]. By treating this problem as a multi-label classification, I am able to train all the possible status by the information given.

![6](https://user-images.githubusercontent.com/38633055/41062496-7f7926cc-69c5-11e8-9deb-f78eda467c1c.PNG)

## 
