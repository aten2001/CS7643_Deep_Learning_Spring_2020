## Self Driving Car Behavioral Cloning ##

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./Behavioral_Cloning_CNN_Architecture.png "Model Architecture"
[image2]: ./examples/left_example.jpg "Left Image"
[image3]: ./examples/center_example.jpg "Center Image"
[image4]: ./examples/right_example.jpg "Right Image"
[image5]: ./examples/final_driving.gif "Final driving"

### Model Architecture and Training Strategy

#### An appropriate model architecture has been employed

This model is based on ['End-to-End Deep Learning for Self-Driving Cars'](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) CNN Network architecture.
1. Keras lambda layer is used to normalized the images, cropping layer is applied to reduce the useless section of training images.
2. 5X5 and 3X3 filters are applied into the Network.
3. Relu activation function is used in convlution layers.

#### Attempts to reduce overfitting in the model

1. Noticed from a training with 50 epoches, the validation loss is stop decrease start from epoch 3. It's obviously the model is orverfitting with too many training epoches.
So the final epoches is 2.
2. Dropout is not tried in this model since final model can drivie the simulator car very well.

#### Model parameter tuning

1. To train the model, the following parameters are used.

| Type         		       | Description	        					                                      |
|:---------------------:|:---------------------------------------------------------------------:|
| Optimizer         		| Keras built in [Adam Optimizer](https://arxiv.org/abs/1412.6980)      |
| Batch Size     	      | 32	                                                                  |
| Epochs					      |	2										                                                  |
| Learning Rate         | 0.001     

#### Appropriate training data
  
1. Images Augmentation  
  a. Flip image, inside the training data/validation data generator, all the images will have a chance to be flipped into a new image. Details can be found at line 24-31 in model.py

2. Data preprocess  
  a. Normalizaiton X/255 - 0.5  
  b. Images cropping  
     Since top area in all driving images is not useful for recognize the track shape. So images are cropped at (0, 75, 0, 25) 
  c. Images drop out, since the training data has a lot of images with steering around zeros. We should droo out some of these 
     images to avoid the model being biased by driving straight

3. Example Training Images  
   The follwing pictures are from the Left, Center, Right camera. From the Training images, we can see our goal is to make the    model can detect the shape of front track and predict the steering angle in order to keep car on the track.
   
   ![Left Iamge][Image2] ![Center Image][Image3] ![Right Image][Image4]
#### Solution Design Approach

The following is the steps to approaching the final model architecture.  
1. First, one round of driving data is collected, a simple linear model is used to try the automonous driving mode, found the simulator car drive off road easily. Obviously, we need more training data and more powerful nerual networks.  
2. 2 More rounds of driving at the center of track are collected, 1 round of reverse direction drviing data is collected.
3. With the suggestion from course, [NVIDIA self driving car CNN archtitecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) is implemented, input image size and output neuros are modified.  
4. Training with the model, found the car can drive smoothly on staright direction. The model tend to drive straight forward with too many zero steering labels.  
5. More Data is collected while driving on curves.  
6. Model still drive off road on curves. We should drop out part of zeros steering images.  
7. 70% of the images which steering is between -0.08 ~ 0.08 is dropped out.   

After these steps, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

The following picture shows the finally model architecture.
![Model architecture][image1]

#### Creation of the Training Set & Training Process

1. Training Data Collection  
  a. left, center, right camera images are used   
  b. drive 3 round nicely at the center of road   
  c. drive 1 round on reverse direction nicely at center of road  
  d. drive on all the curves of track multiple times   
  e. drive off road and drag back to center of road multiple times  

#### Model Automonous Driving
The following gif is a driving shot from final automonous driving mode. More details in driving_video.mp4

![Final Driving][Image5]
