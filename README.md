# GeoSpottr: Image-Based Geolocalization

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Introduction

GeoGuessr is an online game where players are shown a random street view image from anywhere in the world and have to guess the location. This project aims to develop a model that can predict the city in which an image was taken by analyzing visual features extracted from the street view images. By utilizing the power of Convolutional Neural Networks (CNN) and the ResNet50 architecture, we can train a highly accurate image classifier.

![alt text](https://thegatewithbriancohen.com/wp-content/uploads/2016/12/Screen-Shot-2016-12-23-at-10.48.29-PM.png)

## Data Collection

To train our image city predictor model, we collected a large dataset of street view images from various cities around the world using the Google Street View API. The images were labeled by their corresponding city and geocoordinates were also saved for further usage. The dataset encompasses a diverse range of urban landscapes, capturing different architectural styles, landmarks, and environmental characteristics.

Due to time and computational constraints, only 27500 images were collected for 22 randomly selected global cities (1250 samples each).

## Model Architecture

For this project, we utilized the ResNet50 architecture, a deep network that has demonstrated outstanding performance in image classification tasks. ResNet50 is a 50-layer deep neural network that uses residual connections to address the vanishing gradient problem, allowing for the training of very deep neural networks. The weights in the first three stages of the ResNet50 model were frozen, whilst the weights within the very last stage were fine-tuned throughout the training process.

![alt text](https://github.com/pkardjian/GeoSpottr/blob/main/primary/architecture/ResNet.png)

Another fully customized CNN was attached to the end of the ResNet50 model which follows a similar architecture to ResNet-18 where the four main convolutional layers are layered sequentially four times. However, in order to pass the outputs of the pre-trained ResNet-50 (1x1x2048 image tensor) to this part of the model, we must increase the spatial size of the feature maps. Thus, a key component of this model is its input layer which is a transposed convolutional layer. 

![alt text](https://github.com/pkardjian/GeoSpottr/blob/main/primary/architecture/CustomizedCNN.png)

The final part of this model consists of a fully connected artificial neural network (ANN). This part of the model is better known as the head classifier which takes the final output features extracted by the convolutional layers and use them to classify the images. 

## Training

The implementation and training of our model was completed using Python, most notably making use of the PyTorch Torchvision libraries. 

The training process involved fine-tuning the model's weights using the Cross-Entropy loss function and Stochastic Gradient Descent (SGD) optimizer. Training, testing and validation sets were created using a 60-20-20 split to monitor the model's performance and prevent overfitting. Extensive hyperparameter tuning was conducted in order to ensure optimal performance.

## Results

The trained model's performance was evaluated on a separate test dataset, which consisted of street view images from cities not seen during training. The evaluation metrics used include accuracy, precision, recall, and F1 score, providing insights into the model's ability to correctly predict the city in which an image was taken. Additionally, the model's performance was compared with human players in the GeoGuessr game to assess its competitiveness.
