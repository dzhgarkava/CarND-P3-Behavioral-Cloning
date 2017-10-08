**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/left.jpg "Left"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/righr.jpg "Right"

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.py for creating video file for report
* run.mp4 video file with one lap in autonomous mode

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on architecture from NVidia (model.py lines 60-82).
This neural network contains:
* Normalization layer
* 5 Convolutional layers
* 4 Fully-connected layers

Every convolutional layer includes RELU layers to introduce nonlinearity (code lines 64, 66, 68, 69, 70), and the data is normalized in the model using a Keras lambda layer (code line 62). 

####2. Attempts to reduce overfitting in the model

The model contains pooling layers in order to reduce overfitting (model.py lines 65, 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 53-58). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 77).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the model in the lecture. I chose this model just to check that environment works fine.

After that I implemented model with architecture from NVidia.

In order to undestand how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I modified the model and added two pooling layers. It allows to reduce overfitting and still keep model fast for training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track: brifge and two last turns. To improve the driving behavior in these cases, I added more training laps.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 60-82) consisted of a convolution neural network with the following layers and layer sizes:
* Lambda and Cropping layer for image preprocessing (model.py lines 62-63)
* 3 Convolution layers with 5x5 filter sizes and depths between 24 and 48 (model.py lines 64, 66, 68)
* 2 Convolution layers with 3x3 filter sizes and depth 64 (model.py lines 69-70) 
* 4 Fully-connected layers (model.py lines 72-75) 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. Then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to move to the center line. Images from left, center and right cameras look like:

![alt text][image1]
![alt text][image2]
![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would help to train model and add more training examples.

After the collection process and preprocessing this data by flipping images, I had more than 40k number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Video

Attached video shows successful attempt of one lap driving.
