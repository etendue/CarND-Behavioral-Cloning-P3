# *Behavioral Cloning*


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/example1.jpg "Example Image"
[image3]: ./examples/recovery_left.jpg "Recovery Image left"
[image4]: ./examples/recover_right.jpg "Recovery Image right"
[image5]: ./examples/orignal.jpg    "Normal Image"
[image6]: ./examples/flipped.jpg "Flipped Image"
[image7]: ./examples/data_distribution.png "count vs steering angle"
[image8]: ./examples/data_distribution_adapted.png "count vs steering angle"

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

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is built based on reference network published by Nvidia 
(see link [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)).
The  number of layers and parameters are adapted during experimenting with test data. 

It consists of 5 convolution layers (line 148 - 160) and 3 fully connected layers (line 166 - 175).
The convolution layers uses RELU as activation method to introduce non-linearity and the data is normalized in the model 
using a Keras lambda layer (code line 144). 


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 169). 

Train and validation data set are splitted with ratio 4:1, i.e. 80% train data
and 20% validation data. The train uses **EarlyStopping** Callback (see line 184) to control the train process.
```python
early_stop = EarlyStopping(monitor='val_loss', patience=2)
```

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 178).

#### 4. Appropriate training data

Training data was chosed partly from the data provided by Udacity and additionally collected myself using simulator.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My strategy was trying easy and going complicated. 
- I first built the model using only fully connected layer.
- Add the convolution layers
- take the reference model from Nvidia

The original [Nvidia](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
was quickly overfitted during train, so I removed 1 fully connected layer and tuned 
the parameters. Additionally add dropout layer to combat the overfitting.

The model was trained fine and the loss converged quickly without any problem. 
But the performance of model was quite unsatisfied. I moved my focus on the data collection
and pre-processing.


#### 2. Final Model Architecture

The final model architecture (model.py lines 134-175) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image           				|
| Cropping2D            | input=160x320x3 output=90x320x3               |
| Normalizing/Lambda    | Pixel values (0~255) to (-1.0 ~ 1.0)          |
| Conv2D(24) 5x5 relu 	| 2x2 stride, valid padding, outputs=43x158x24	|
| Conv2D(36) 5x5 relu 	| 2x2 stride, valid padding, outputs=20x77x36	|
| Conv2D(48) 5x5 relu 	| 2x2 stride, valid padding, outputs=8x37x48	|
| Conv2D(64) 3x3 relu 	| 1x1 stride, valid padding, outputs=6x35x64	|
| Conv2D(64) 3x3 relu 	| 1x1 stride, valid padding, outputs=4x33x64	|
| Flatten               | output 8448                                   |
| Dense(256)    		| output 256  							        |
| Dropout        		| keep_prob = 0.5 output 256       	            |
| Dense(64)     		| output 64  							        |
| Dense(1)      		| output 1  							        |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle
would learn to steer back to center of road. These images show what a recovery looks like starting from the edges of roads.

![alt text][image3] ![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add training data. 
For example, here is an image that has then been flipped:

![alt text][image5] ![alt text][image6]

Etc ....

After the collection process, I had c.a 7,000 number of data points.
I then expands this data by 
1. using images from left and right cameras by add steering angle corrections.
```python
# add correction, using left and right cameras
    df_left["angle"] += 0.2
    df_right["angle"] -= 0.2
```
2. flipping all the images 

The data processing is encapsulated in function (line 9 - 82):
```python
def prepare_data():
```

Finally It is more than 42,000 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 13 though I set epochs to 20. The train stopped by EarlyStopping callback.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

### !!!However!!!
by all these efforts the model did not work well. In contrast it was even worse in comparision
to no augmenting the data. Why?

I inspected every layer to check if there is anything wrong also in function
prepare_data(), if everything works as expected. I did research and one idea hit me
that a good train data needs balanced distribution. The issue was also discussed in 
***Project 2 CarND-Traffic-Sign-Classifier-Project***. 
Due to unbalanced train data vs classes, the model performs differently on different 
traffic classes.

I inspected the "data count" vs "steering angle".
Here is the original distribution.

![alt text][image7].

The data set with steering angle ~ 0 is dominant; the other two spikes are introduced by using
left, right cameras, i.e. the steering angle are shifted by +/- 0.2 correction.
This introduces high bias to stay the steer wheel to 0 and is not wanted.

I removed the peaks by fit the data with **scipy.interpolate** and cut the peaks.
Here is  the more reasonable distribution.

![alt text][image8].


Finally the model produces expected performance.