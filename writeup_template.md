# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://devblogs.nvidia.com/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center.jpg "Center Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 - A video recording of your vehicle driving autonomously at least one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes, strides 2 and depths between 24 and 48 (model.py lines 69-71) 
My model consists of a convolution neural network with 3x3 filter sizes and depth 64 (model.py lines 72-73)

The model includes RELU layers to introduce nonlinearity (code lines 69 - 73), and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 84--89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also left side of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive as much as I can drive with simulator. Also I did rtain model on some sharp returns.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because we used this to recognize german traffic lights.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with dropout in order to reduce overfitting parameters.

The model used an adam optimizer, so the learning rate was not tuned manually

Then I implemented Nvidia architecture instead of LeNet to get more accurate results.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specially on sharp turns and to improve the driving behavior in these cases, I drive few more laps and try to collect data specially on these spots

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 65-79) consisted of a convolution neural network with the following layers and layer sizes;
```python
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5, strides=(2,2),activation='relu'))
model.add(Conv2D(36,5, strides=(2,2),activation='relu'))
model.add(Conv2D(48,5, strides=(2,2),activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]


Then I repeated this process on track two in order to get more data points.I also did reverse track in order to collect more data.

To augment the data sat, I also flipped images and angles thinking that this would increase my training data and generate better model.I also use ;eft and right camera images wirh correction rates 0.2 to angle , in order to recover my car when it goes outside of model.


After the collection process, I had 5730*3*2=34380 number of data points. I then preprocessed this data by cropping 70px from top and 25px fro bottom to increase calculation speed.


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by my model. I used an adam optimizer so that manually training the learning rate wasn't necessary.
