
# **Traffic Sign Recognition** 
---

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./saved_images/trainingDataset.png "Traffic Signals from Training Data Set"
[image2]: ./saved_images/distribution_training_set.png "Distribution of images in Training Data Set"
[image3]: ./saved_images/preprocessed.png "Preprocessing"
[image4]: ./saved_images/german_websigns.png "Traffic Signs"

---
#### Project Code

This is the link to my [project code](./Traffic_Sign_Classifier.ipynb)

### ** Data Set Summary & Exploration**

#### 1. Basic Summary of data set

I have used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
Below are some traffic sign images from training data set.


![alt text][image1]

The bar chart showing the image data distribution of the training data. Each bar represents one class (traffic sign) and number of images in the class. The mapping of traffic sign names to class id can be found here:[signnames](./signnames.csv)
![alt text][image2]

Similar visualization can be found in code block 7 & 8 for Validation and Test data set.

### **Design and Test a Model Architecture**

#### 1. Preprocessing
I have applied two processing techniques.Refer to code block 9 for implementation of preprocessing techniques. 
* Conversion to Grayscale : The images in dataset are in RGB format. The color of traffic signs are not required to detect it.Moreover,reducing from 3channel to single channel will reduce the amount of input data and model training time.It is helpful when GPU is not available.


* Normalization of image : Normalization is required as the dataset has wider distribution and it will be difficult to train the model with single learning rate.It helps in training the model faster.

Here is an example of a traffic sign image after grayscaling and normalization.
![alt text][image3] 

#### 2. Model Architecture
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|
| Input         		| Grayscale image  							    |
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Same padding , outputs 16x16x6 	|
| Convolution 5x5	    | 1x1 stride, Valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, Valid padding , outputs 5x5x6 	|
| Flatten               | Output 400                                    |
| Fully connected		| Output 120     								|
| RELU					|			 								    |
| Dropout               | Keep_probability - 0.85                       |
| Fully connected		| Output 84     								|
| RELU					|			 								    |
| Dropout               | Keep_probability - 0.85                       |
| Fully connected		| Output 43     								|
 
#### 3. Training model

To train the model, I have used the following parameters :
* Batch Size - 128
* Epoch - 30
* Learning Rate - 0.001
* Dropout - 0.85
* Optimizer - Adam optimizer

Dropout is used only for training the model.For validation and test dataset,keep probability is 1.

#### 4. Approach to get validation accuracy above 0.93

**My final model results were:**
* training set accuracy of 0.999
* validation set accuracy of 0.944 
* test set accuracy of 0.920

**Adjustments to LeNet-5 Architecture:**

* The preprocessing of images helped in training the model faster and accurately.Before preprocessing,validation accuracy was 0.86 and after preprocessing on epoch 10, training accuracy was 0.98 and validation accuracy was 0.911 and test accuracy 0.891.
* Increased epoch from 10 to 30.This was based on trial and error method.Higher accuracy was when epoch was 30.
* On applying dropout regularization of 0.85 and epoch 30,the validation accuracy improved significantly to 0.944. Dropout helped in preventing overfitting of the model.Before using dropout,the training accuracy was 0.997 but the validation accuracy was 0.922. Therefore,Dropout is one of the important technique to increase accuracy of the model and reduce overfitting.

 

### Test a Model on New Images

#### 1. German traffic signs found on the web .
Here are six German traffic signs that I found on the web:

![alt text][image4]

The Image 1 might be difficult to classify because of noisy background.  
The Image 2 might be difficult to classify because of noisy background.  
The Image 3 might be difficult to classify because it's rotated.  
The Image 4 might be difficult to classify because it's rotated.  
The Image 5 might be difficult to classify because it's rotated and background has 2 parts.  
The Image 5 might be difficult to classify because it's rotated.  

#### 2. Model's predictions on new traffic signs 
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)                          |
| Keep Right            | Keep Right                                    |
| Pedestrians           | Pedestrians                                   |
| Priority Road         | Yield                                         |
| Slippery Road         | Speed limit (30km/h)                          |
| Stop      		    | Stop       									| 

Accuracy = 66.7%
The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This compared to the accuracy on the test set of 92.0 is very low but with such a small set of images we can not calculate a good accuracy.

#### 3. The top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 21st cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit(30km/h) sign (probability of 1.0), and the image does contain a Speed limit (30km/h)    sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Speed limit (30km/h)                          | 
| 0.0     				| Speed limit (70km/h)                          |
| 0.0					| Speed limit (20km/h)                          |
| 0.0	      			| General caution                               |
| 0.0				    | Roundabout mandatory                          |


For the second image,the model is relatively sure that this is a Keep Right sign (probability of 1.0), and the image does contain a Keep Right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Keep Right   									| 
| 0.0     				| Turn left ahead						        |
| 0.0					| Beware of ice/snow				     		|
| 0.0	      			| Roundabout mandatory			 				|
| 0.0				    | Right-of-way at the next intersection			|

For the third image,the model is relatively sure that this is a Pedestrians sign (probability of 0.99), and the image does contain a Pedestrians sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         		    | Pedestrians    				     			| 
| .000     			    | Right-of-way at the next intersection			|
| .000				    | Double curve						         	|
| .000	      		    | Road narrows on the right			            |
| .000				    | General caution        						|

For the fourth image,the model is incorrect that this is a Yield sign (probability of 0.99).The image is a Priority Road sign(probability of 0.39). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         		    | Yield   					     		    	| 
| .000    			    | No entry 						        	    |
| .000				    | Priority road									|
| .000	      		    | Turn right ahead				 				|
| .000				    | Ahead only      						    	|

For the fifth image,the model is incorrect that this is a Speed limit (30km/h) sign (probability of 0.98), The image is a Slippery road sign(probability of 0.000). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .981      			| Speed limit (30km/h)     						| 
| .018   				| Roundabout mandatory		    				|
| .000					| Double curve				         			|
| .000	      			| Speed limit (70km/h)				    		|
| .000				    | Road narrows on the right         		    |

For the sixth image,the model is relatively sure that this is a Stop sign (probability of 1.0), and the image does contain a Stop sign.The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Stop  									    | 
| .000   				| Keep left 						    		|
| .000					| Yield							        		|
| .000	      			| Speed limit (50km/h)					 	    |
| .000				    | Speed limit (30km/h)     						|

