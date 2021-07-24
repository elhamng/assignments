## Detection Algorithms

### Object Localization
Image classification takes an image and classify it in a given set of classes. Classification with localization also pinpoint the bound box of the localization of the object in the image. The object detection usually works out with multiple tasks at the same time.

Localization can be taught to the network with the output parameters being set as b_x,b_y for the bounding box middle point and b_w,b_h for the width and height respectively. Therefore the output of the network now has 4 extra outputs which determine the position of the object in the given image. To help the learning the values are normalized using one of the corners (as the top left) as (0,0) and the opposite corner as (1,1).

The loss here is only calculated when an object is detected. When no object is detected we don’t care for the bounding box positions calculated and those are not taken into the loss of that particular step.

### Landmark Detection
Landmark detection basically is the subset of localization where you pinpoint the localization of several points in the image at the same time, like in the Snapchat filter where it pinpoints the faces and uses that input to apply a filter.

###Object Detection
Using slide window detection you can build a ConvNet that detects a given object using a small sample of image and use a sliding window to classify over a bigger image. Given different sizes and strides of sliding windows you can detect the position of objects at the cost of high computational cost if the windows are sequentially processed.

Sliding Windows Detection :
you take these windows, these square boxes, and slide them across the entire image and classify every square region with some stride, a huge disadvantage of Sliding Windows Detection is the computational cost. 

Convolutional Implementation of Sliding Windows:
Turning FC layer into convolutional layers
Instead of doing it sequentially, with this convolutional implementatio, you can implement the entire image, all maybe 28 by 28 and convolutionally make all the predictions at the same time by one forward pass through this big convnet.

## What is Intersection over Union(IoU)? 
Intersection over Union is an evaluation metric used to measure the accuracy of an object detector on a particular dataset. 
Intersection over Union uses to evaluate the performance of HOG + Linear SVM object detectors and Convolutional Neural Network detectors (R-CNN, Faster R-CNN, YOLO, etc.).


![image](https://user-images.githubusercontent.com/64529936/123420306-b2c83d00-d5bb-11eb-951a-ed63ebbd8ae4.png)

In the numerator we compute the area of overlap between the predicted bounding box and the ground-truth bounding box.

The denominator is the area of union, or more simply, the area encompassed by both the predicted bounding box and the ground-truth bounding box.

Dividing the area of overlap by the area of union yields our final score — the Intersection over Union.
An Intersection over Union score > 0.5 is normally considered a “good” prediction.

### Anchor boxes
Define some anchor boxes, for example, two: one wider and one taller. Now your output will double as you want to predict each anchor box separately. The previous output was y=[pc,bx,by,bh,bw,c1,c2,c3] for a given set of 3 classes. The new output will be y=[pc,bx,by,bh,bw,c1,c2,c3,pc,bx,by,bh,bw,c1,c2,c3] where each half will be interpreted as separates bounding boxes.

### YOLO algorithm
y has shape n_cells−width × n_cells−height × n_bounding−boxes × (n_classes+5). The 5 in the number of classes comes from the pc,bx,by,bh,bw terms.

non-max suppression
As the grid gets finer it’s possible that multiple cells detect the object on them and end up firing the detection of the bounding box on mutiple places. The non-max supression technique chooses only the highest bounding box in the classification to output as result.



Deep learning for semantic segmentation 
U-net Architecture : one of the most important and foundational neural network architectures of computer vision today


## Convolution operation
There are two inputs to a convolutional operation
A 3D volume (input image) of size (nin x nin x channels)

A set of ‘k’ filters (also called as kernels or feature extractors) each one of size (f x f x channels), where f is typically 3 or 5.


The output of a convolutional operation is also a 3D volume (also called as output image or feature map) of size (nout x nout x k).

![image](https://user-images.githubusercontent.com/64529936/123611754-370c0180-d802-11eb-81af-88b4794b0673.png)

receptive field (context) is the area of the input image that the filter covers at any given point of time.

Max pooling operation
the function of pooling is to reduce the size of the feature map so that we have fewer parameters in the network.

A very important point to note here is that both convolution operation and specially the pooling operation reduce the size of the image. This is called as down sampling.

In a typical convolutional network, the height and width of the image gradually reduces (down sampling, because of pooling) which helps the filters in the deeper layers to focus on a larger receptive field (context).the number of channels/depth (number of filters used) gradually increase which helps to extract more complex features from the image.

the output of semantic segmentation is not just a class label or some bounding box parameters. In-fact the output is a complete high resolution image in which all the pixels are classified.

transposed convolution the input volume is a low resolution image and the output volume is a high resolution image.

one shot learning problem ?





















