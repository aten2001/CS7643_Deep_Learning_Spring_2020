## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./readme_images/train_imgs_visualization.png "Visualization"
[image2]: ./readme_images/Sign_type_bar_chart.png "Sign counts"
[image3]: ./readme_images/visualization_featureMaps.png "Feature Map"
[image4]: ./test_images/120_limit.jpg "Traffic Sign 1"
[image5]: ./test_images/70_limit.jpg "Traffic Sign 2"
[image6]: ./test_images/No_entry.jpg "Traffic Sign 3"
[image7]: ./test_images/Yield.jpg "Traffic Sign 4"
[image8]: ./test_images/stop.jpg "Traffic Sign 5"

### Data Set Summary & Exploration

#### 1. Summary of data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

| Type         		       |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Training Set       		 | 34799   							 | 
| Validation Set     	  | 4410 	          |
| Test Set					         |	12630						     |
| Image Shape	      	   | (32, 32, 3) 				|
| Unique Classes        | 43      								|

#### 2. Visualization of data set

Here is an exploratory visualization of the data set. The data set has 43 class type of traffic sign, each sign has samples range from 200~2000.

![Traffic sign types][image1]
![Traffic sign distribution][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing data

1. Grayscale: since the color of traffic signs is not a decisive facotr of traffic sign types. cv2.cvtColor is used to convert the RGB image to grayscale images.
2. Normalization: to avoid optimizer doing a lot o searching to find optimized solution, (x - 128.0)/255.0 is used to normalized all the pixel sacle to [-0.5, 0.5]


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		      |     Description	        					                 | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 RGB image   							                   | 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x16 	 |
| RELU					             |												                                   |
| DROPOUT               | 0.8                                           |
| Max pooling	      	   | 2x2 stride,  outputs 14x14x16 				            |
| Convolution 5x5	      | 1X1 stride, valid padding, outputs 14x14X48   |
| RELU                  |                                               |
| DROPOUT               | 0.6                                           |
| Max pooling           | 2X2 stride, outputs 5X5X48                    |
| Flateen               | output 1200                                   |
| Fully connected		     | output 240        									                   |
| Fully connected       | output 96                                     |
| RELU                  |                                               |
| Fully connected       | output 43                                     |
 

#### 3. Model Training Paramters

To train the model, I used an following optimizer and parameters:

| Type         		       | Description	        					                                             |
|:---------------------:|:---------------------------------------------------------------------:|
| Optimizer         		  | Tensorflow built in [Adam Optimizer](https://arxiv.org/abs/1412.6980) |
| Batch Size     	      | 128	                                                                  |
| Epochs					           |	40										                                                          |
| Learning Rate         | 0.001                                                                 |


#### 4. Approach to a solution

My final model results were:

| Data set               | Accuracy         |
|:----------------------:|:----------------:|
| Validation Set         | 94.7%            |
| Test Set               | 93.2%            |

Model architecure chosen approach:
1. First i used the LeNet and change input to 32X32X3, the validation reuslt is less than 90%.
2. Then I am adding a dropout = 0.8 after first convoluation layer, the validation increase a bit, but still less than 93%.
3. Since this data set has 43 classes which large than MINIST 10 types, i increased neuro number at each layers. After 200 epochs at learning rate 0.001, the valiation accuracy went up to 95% after epochs 30. Then drop down to 93.5%.
4. So it's probablly overfitting, then another dropout=0.6 is added to second convoluation layer. And decrease epochs to 40.
5. Finally, the validation accuracy is 94.7% and test accuracy is 93.2%.

Why Choose LeNet as the initial architecture:
1. LeNet is proved to have an accruacy of 99% for MINIST
2. LeNet's layers has the ability to recoginize handwriting shapes and edges. This is similiar to traffic sign classification.
To recognize traffic signs, the shape and edges are core factors.
So choosing LeNet as an initial architecture and then finetuning the architecture works well for this project.

### Evaluate The Model

#### 1. Downloaded 5 German traffic signs on google images.

Here are five German traffic signs that I found on the web:

![120 KM/h Speed Limit][image4] 
![70 KM/h Speed Limit][image5] 
![No Entry][image6] 
![STOP Sign][image7]  
![Yield][image8] 

Here are the results of the prediction:

| Image			              |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| 120 KM/h Speed Limit  | 50 KM/h Speed Limit   									               | 
| 70 KM/h Speed Limit   | 120 KM/h Speed Limit 										               |
| No Entry              | No Entry                           											|
| STOP Sign	      		    | STOP Sign					 				                           |
| Yield               		| Yield                                  							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.
Comparing to test data set accuracy 93.2%, this is relatively low. The probelms could be:
a. Model doesn't have a strong ability to recognize the numbers in traffic signs
b. Image background, angles, traffic sign position in the image are quite different from training data set.

#### 3. Prediction probabilities

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability			        |     Prediction	        					                  | 
|:---------------------:|:---------------------------------------------:| 
| 0.649                 | No Passing            									               | 
| 0.528                 | Roundabout mandatory 										               |
| 1.00                  | No Entry                           											|
| 0.99	      		         | STOP Sign					 				                           |
| 1.00               		 | Yield                                  							|

### Visualizing the Neural Network 
The following picture shows the feature maps of first convoluation network.
As we can see, the shapes of sign and STOP characters are recoginized. 
![Feature Mpas][image3]

