# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md  summarizing the results
* video.mp4 recorded video

**To run the pretrained model**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

**To train the model**
```sh
python model.py
```
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

The model architecture is based on [The NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) which is a deep convolution network.
It works very well with supervised image classification problem.

The model has a Keras lambda layer to mormalize input image and ELU for activation function for every layer are included except for the output layer to introduce non-linearity. A dropout layer is added to avoid overfitting after the convolution layers.

| Layer (type)                   | Description                                    |
|--------------------------------|------------------------------------------------|
|lambda_1 (Lambda)               | Image normalization                            |
|convolution2d_1 (Convolution2D) | 5x5, filter: 24, strides: 2x2, activation: ELU |
|convolution2d_2 (Convolution2D) | 5x5, filter: 36, strides: 2x2, activation: ELU |
|convolution2d_3 (Convolution2D) | 5x5, filter: 48, strides: 2x2, activation: ELU |
|convolution2d_4 (Convolution2D) | 3x3, filter: 64, strides: 1x1, activation: ELU |
|convolution2d_5 (Convolution2D) | 3x3, filter: 64, strides: 1x1, activation: ELU |
|dropout_1 (Dropout)             | Drop out (0.5)                                 |
|flatten_1 (Flatten)             | neurons: 100, activation: ELU                  |
|dense_1 (Dense)                 | Fully connected: neurons: 50, activation: ELU  |
|dense_2 (Dense)                 | Fully connected: neurons: 10, activation: ELU  |
|dense_3 (Dense)                 | Fully connected: neurons: 10, activation: ELU  |
|dense_4 (Dense)                 | Output                                         |




### Preprocessing Data
  (1): Image modification<br>
    1: crop image top and bottom, because the sky and the car front is not necessary for the model.<br>
    2: resize to 66x200 (3 YUV channels) as per NVIDIA model<br>
    3: In order to avoid saturation affect the training and make gradients work better, the images are normalized
    by dividing pixel value by 127.5 and subtracted 1.0<br>

  (2)Image Augumentation<br>
    * Randomly select right, left or center images.<br>
    * flip image left/right<br>
    * translate image horizontally with steering angle adjustment (0.002 per pixel shift)<br>
    * translate image virtically<br>
    * altering image brightness <br>


### Training, Validation and Test
For training, the loss function is mean squared error which can measure how close the model predicts to the given streeing angle in each images. the learning rate is 1.0e-4. the optimization is Adam optimizer.

The images are splitted into train and validation set. Testing was done with the simulator.
