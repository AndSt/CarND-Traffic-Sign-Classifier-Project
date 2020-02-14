# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how often each label occurs. Each label corresponds to a specific type of traffic sign.

![alt text][./data/barplot.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing consists of two steps. 

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][examples/grayscale.jpg]

As a second step, I normalized the image data because it has been proven to be extremely helpful for neural architectures. A main concern are the exploding and vanishing gradient problem which are reduced by normalized data.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I was googling around a lot to see what works on the Data set. I found https://chsasank.github.io/keras-tutorial.html to be very useful and built up on it. I played around with preprocessing, channel size, dropout rate and batch normalization. In conclusion I think the model is still way undercapacitated. In comparison to state-of-the-art CV models it can be considered tiny, indicating that the inductive bias (approximator definition) is far from ideal for the classification. Then a more rigorous regularization would be needed.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Dropout			    | Rate 0.2     									| 
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Dropout			    | Rate 0.2     									| 
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Convolution 3x3     	| 3x3 stride, same padding, outputs 32x32x128 	|
| RELU					|												|
| Dropout			    | Rate 0.2     									|
| Flatten			    | Rate 0.2     									|
| Fully connected		| Units 512    									|
| Dropout			    | Rate 0.2     									|
| Fully connected		| Units 256    									|
| Dropout			    | Rate 0.2     									|
| Softmax				| 43 classes   									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the Adam Optimizer with a learning rate of 0.1 and a batch size of 128 and trained for 10 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9914
* validation set accuracy of 0.9819
* test set accuracy of 0.9633

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
https://chsasank.github.io/keras-tutorial.html
* Why did you believe it would be relevant to the traffic sign application?
Because it already shows good results even though it overfitted a bit.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 I changed the model slightly. I played around with dense sizes, dropout values and added another linear layer.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![50km][data/dowloaded_images/50_sign.jpg] ![Children sign][data/dowloaded_images/children_sign.jpg] ![Roundabout][data/dowloaded_images/roundabout_sign.jpg] 
![Stop Sign][data/dowloaded_images/stop_sign.jpg] ![Yield][data/dowloaded_images/yellow_sign.jpg]

Most of there are not taken from a real world use case, but are drawn via computer. Thus the scenario is different which should pose problems for the classifier. Specifically, the white background might be difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 Sign      			| 20 sign   									| 
| Children     			| General caution 								|
| Roundabout     		| Priority road 								|
| Stop Sign      		| Stop sign					 					|
| Yield Sign			| Ahead only      								|


Accuracy is 20%. This is a huge difference from the test set. That was kind of expected as the images are so different in terms of their surroundings and occurences.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the fourth image, the model is pretty sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| <.01     				| No entry 										|
| <.01					| Keep left										|
| <.01	      			| 80 sign   					 				|
| <.01				    | 60 sign           							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


