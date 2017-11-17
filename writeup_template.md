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

[image1]: ./examples/original.png "Original image"
[image2]: ./examples/gray.png "Gray image"
[image3]: ./examples/augmented.png "Augmented image"
[image4]: ./real/1.png "4th image" 
[image5]: ./real/2.png "5th image" 
[image6]: ./real/3.png "6th image" 
[image7]: ./real/4.png "7th image" 
[image8]: ./real/5.png "8th image" 
[image9]: ./real/gray/16.png  "9th image" 
[image10]: ./real/gray/17.png "10th image" 
[image11]: ./real/gray/18.png "11th image" 
[image12]: ./real/gray/19.png "12th image" 
[image13]: ./real/gray/20.png "13th image" 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used length and shape methods to get the summary data.

Here are the summary of the dataset.

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43
 
#### 2. Include an exploratory visualization of the dataset.

I used matplot library to verify the image before further proceedings. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I used LeNet architecture and that has only 60K paramaters and may not fit well for the color images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then, I normalized the image data because it provides the data ranges from 0 to 1 so learning would be faster and the gradient would not fluctuate much. I used the technique same like Tensorflow, I copied the tensorflow base python code for this purpose. It calculates the mean and SD and then apply the normalization formula.

At last step, I applied image augmentation (I would explain in detail below,  how it helped me to get higher accuracy). I used package called 'imgaug' and it was simple to follow. I applied the below augmentation. 
   
* horizontal flip
* Vertical flip
* image blur
* image translate
* scaling
* dropout
* coarse dropout
* invert color channelsut
* change brightness

Here is an example of an augmented image:

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer            | Description                                 |
| -------------    | -------------                               |
| Input            | 32x32x1 grayscale image                     |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU             |                                             |
| Max pooling      | 2x2 stride, outputs 14x14x6                 |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU             |                                             |
| Max pooling      | 2x2 stride                                  |
| Flatten          | 400                                         |
| Fully connected  | input - 400, output - 120                   |
| RELU             |                                             |
| Dropout          | 0.5 keep                                    |
| Fully connected  | input - 120, output - 84                    |
| RELU             |                                             |
| Dropout          | 0.5 keep                                    |
| Fully connected  | input - 84, output - 43                     |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam optimizer with learning rate '0.00097', 30 epochs, xavier initialization and a batch size 156.

I changed the parameters based only on accuracy and not training time.

I will explain the more about the performance details in the next cell. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

* What architecture was chosen?

I choosed the LeNet architecture, I wanted to stick with 5 layer net so I have more chance to optimize the parameters, also I can think about various augmentation to increase the accuracy.

* Why did you believe it would be relevant to the traffic sign application?

When I ran Lenet architecture first, I got more than 85% of accuracy that's give me a confident that I could stay with basic architecture and tune the parameters, provide more augmented images to reach 93% accuracy.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I got a test accuracy of 100% and a validation accuracy of 95%. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I choose 5 images from the web, converted the images into grayscale and normalized before prediction.

Here are five German traffic signs that I found on the web:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Model results showed that it did't fluctuate and generalized well. Minimum top guess is 82%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


