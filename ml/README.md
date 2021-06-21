## What is machine learning?

Arthur Samuel defined machine learning as the field of study that gives computers the ability to learn without being explicitly learned.
Tom Mitchell says, a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

## Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

## Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

## Cost Function
We can measure the accuracy of our hypothesis function by using a cost function. 
 Hypothesis function and cost function in linear regression  
![image](https://user-images.githubusercontent.com/64529936/119698074-29eda280-be51-11eb-8426-9e4cb7715cea.png)

## Hypothesis in linear regression  
<img src="https://render.githubusercontent.com/render/math?math=h(\theta) = \theta_1 x %2B \theta_0">
 
 in many features 
 
 <img src="https://render.githubusercontent.com/render/math?math=h(\theta) = \theta_1 x_1 %2B \theta_2 x_2 %2B \theta_3 x_3 %2B...%2B (\theta_0">
 
 ## Feature scaling 
 ideat that make sure features are on a similar scale.
 
 ## Random Forest
 Random forest is a model that is bult up many decision trees whose features selected random. it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters. 
 Random forest is the most used supervised machine learning algorithm for regression and classification.
 ![image](https://user-images.githubusercontent.com/64529936/122762647-d4fd4a80-d29d-11eb-876e-20e00f6feb0c.png)
 
 There are two main steps for random forest model : 
 
 first create a bootstrapped dataset 
 for example table shows an original dataset
 ![image](https://user-images.githubusercontent.com/64529936/122763168-50f79280-d29e-11eb-9263-9b7faaf80300.png)
 
 We select randomly a subset from original dataset like :
 ![image](https://user-images.githubusercontent.com/64529936/122763332-81d7c780-d29e-11eb-9a08-c55b57946f16.png)
![image](https://user-images.githubusercontent.com/64529936/122763467-a764d100-d29e-11eb-9451-16efb296da75.png)



Step two: Creas an individual tree for each subset
![image](https://user-images.githubusercontent.com/64529936/122767500-07f60d00-d2a3-11eb-9914-13ee7f2f131d.png)

![image](https://user-images.githubusercontent.com/64529936/122767631-26f49f00-d2a3-11eb-9d65-aa3b7e5b264a.png)
repeat for 6 subsets 
![image](https://user-images.githubusercontent.com/64529936/122769295-bcdcf980-d2a4-11eb-9513-e5bea7c81a85.png)


Bootstrapping data and using its aggregate to make a decission is known as bagging, i.e., bagging is training a bunch of individual models parallelly that each model is trained by a random subset of the data. 
The size of data remains the same,  all the subsets are equal in terms of number of rows. So all of the subsets have the same nubmber of rows as the main data.

![image](https://user-images.githubusercontent.com/64529936/122768377-d3cf1c00-d2a3-11eb-876f-5704633bebbe.png)

there is two diffrences between the dataset and the subsets : 1) the size of row the same but data are diffrenet because we select data by random sampling with replacement and pass on this data to different trees where we have different columns for the tree which is at random and then we are going to train these individual trees and then we will take vote of these trees.  
![image](https://user-images.githubusercontent.com/64529936/122767160-a59d0c80-d2a2-11eb-92f2-3c3cc62b85ac.png)

All the steps:

![image](https://user-images.githubusercontent.com/64529936/122772926-26aad280-d2a8-11eb-85e9-6f65eafe384c.png)

![image](https://user-images.githubusercontent.com/64529936/122774693-b4d38880-d2a9-11eb-9b8b-1193de60235c.png)
different nodes take different columns to do splitting data that is why decision trees will be different from each other. 
![image](https://user-images.githubusercontent.com/64529936/122774801-cf0d6680-d2a9-11eb-8499-56e5203e738c.png)

keep doing this untill we have significant number of trees 
![image](https://user-images.githubusercontent.com/64529936/122776348-50192d80-d2ab-11eb-8c27-9d46d15471a8.png)
![image](https://user-images.githubusercontent.com/64529936/122776454-67f0b180-d2ab-11eb-8171-4a9c7fdee9fd.png)

There are some of the trees which have never seen out of bag row








 

 
 

 
    
