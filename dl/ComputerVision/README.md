## What is computer vision ?

Computer Vision (CV) is a subfield of artificial intelligence and machine learning that develops techniques to train computers to interpret and understand the contents inside images.

Images are a made up of thousands of pixels. These pixels are one-dimensional arrays with values from 0 to 255. One single image will contain three different matrices for the three components that represent the three primary colours: red, green and blue (RGB).

By combining different brightness levels of the different primary colours (from 0 to 255), a pixel can display alternate colours to those primary ones.By analysing the brightness values of a pixel and comparing it to its neighbouring pixels a computer vision model is able to identify edges, detect patters and eventually classify and detect objects in an image based on previously learned patterns.

Often, computers require images to be pre-processed prior to applying any detection and tracking models to them. Image pre-processing simplifies and enhances the image’s raw input by changing its properties, such as its brightness, colour, cropping, or reducing noise.

Deep learning models are often trained to automate this process by inputting thousands of pre-processed, labelled or pre-identified images.

Models may also use X and Y coordinates to create bounding boxes and identify everything within each box, such as a football field, an offensive player, a defensive player, a ball and so on.

Its goal is to give computer the ability to extract high-level understanding from digital images and vidoes. 
Images on computers are most often stored as big grids of pixels. Each pixels is defined by a color, stored as a combination of three additive primary colors. 

Data augmentation is the simplest way to reduce overfitting to increase the size of the training data.
In keras we can perform all data transformation using ImageDataGenerator 
## The Applications Of Computer Vision In Sport.
Most major sports involve fast and accurate motion that can sometimes become challenging for coaches and analysts to track and analyse in great detail.

Automated methods for sports video understanding can help in the localization of the salient actions of a game. Detecting soccer actions is a difficult task due to the sparsity of the events within a video.

The angle, positioning, hardware and other filming configurations of these cameras can vary greatly from sport to sport, event to event or even within the different cameras used for the same match or training session.

One of the key aims when applying computer vision in sports is player tracking. 

Many automated sports analytics methods have been developed in the computer vision community
to understand sports broadcasts

analyze semantic information to
automatically detect goals, penalties, corner kicks, and card
events.

## What is object detection? 

Object detection can be defined as a branch of computer vision which deals with the localization and the identification of an object



In the RetinaNet configuration, the smallest anchor box size is 32x32. This means that many objects smaller than this will go undetected.

What is the smallest size box I want to be able to detect?
What is the largest size box I want to be able to detect?
What are the shapes the box can take?

The(X, Y, Height, Width) is called the “bounding box”, or box surrounding the objects.

For each anchor box, calculate which object’s bounding box has the highest overlap divided by non-overlap. This is called Intersection Over Union or IOU.

pre-trained object detection architectures

CenterNet (2019) is an object detection architecture based on a deep convolution neural network trained to detect each object as a triplet (rather than a pair) of keypoints, so as to improve both precision and recall. 

MobileNet is an object detector released in 2017 as an efficient CNN architecture designed for mobile and embedded vision application. This architecture uses proven depth-wise separable convolutions to build lightweight deep neural networks


![image](https://user-images.githubusercontent.com/64529936/125648818-8e6598d1-1f02-41b0-84d5-d4783f69e64c.png)


batch and real-time

blob storage ?