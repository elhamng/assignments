What is computer vision ?
Computer Vision (CV) is a subfield of artificial intelligence and machine learning that develops techniques to train computers to interpret and understand the contents inside images.

Images are a made up of thousands of pixels. These pixels are one-dimensional arrays with values from 0 to 255. One single image will contain three different matrices for the three components that represent the three primary colours: red, green and blue (RGB).

By combining different brightness levels of the different primary colours (from 0 to 255), a pixel can display alternate colours to those primary ones.By analysing the brightness values of a pixel and comparing it to its neighbouring pixels a computer vision model is able to identify edges, detect patters and eventually classify and detect objects in an image based on previously learned patterns.

Often, computers require images to be pre-processed prior to applying any detection and tracking models to them. Image pre-processing simplifies and enhances the image’s raw input by changing its properties, such as its brightness, colour, cropping, or reducing noise.

Deep learning models are often trained to automate this process by inputting thousands of pre-processed, labelled or pre-identified images.

Models may also use X and Y coordinates to create bounding boxes and identify everything within each box, such as a football field, an offensive player, a defensive player, a ball and so on.



Its goal is to give computer the ability to extract high-level understanding from digital images and vidoes. 
Images on computers are most often stored as big grids of pixels. Each pixels is defined by a color, stored as a combination of three additive primary colors. 



data augmentation is the simplest way to reduce overfitting to increase the size of the training data.
In keras we can perform all data transformation using ImageDataGenerator 




