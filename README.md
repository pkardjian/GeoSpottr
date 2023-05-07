# GeoSpottr: Image-Based Geolocalization

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Introduction

GeoGuessr is an online game where players are shown a random street view image from anywhere in the world and have to guess the city or location. This project aims to develop an AI model that can predict the city in which an image was taken by analyzing visual features extracted from the street view images. By utilizing the power of deep learning and the ResNet50 architecture, we can train a highly accurate image classifier.

![alt text](https://www.geoguessr.com/seterra/images/system/gg-teaser.png)

## Data Collection

To train our image city predictor model, we collected a large dataset of street view images from various cities around the world using the Google Street View API. The images were labeled with their corresponding city information. The dataset encompasses a diverse range of urban landscapes, capturing different architectural styles, landmarks, and environmental characteristics.

## Model Architecture

For this project, we employed the ResNet50 architecture, a deep CNN that has demonstrated outstanding performance in image classification tasks. ResNet50 is a 50-layer deep neural network that uses residual connections to address the vanishing gradient problem, allowing for the training of very deep networks. The pre-trained weights of the ResNet50 model, trained on the large-scale ImageNet dataset, were used as a starting point for transfer learning.

## Training

We trained the ResNet50 model on our collected dataset using transfer learning. The pre-trained weights were loaded into the model, and only the final fully connected layer was retrained on our specific task. The training process involved fine-tuning the model's weights using backpropagation and optimizing the parameters with the Adam optimizer. The training dataset was split into training and validation sets to monitor the model's performance and prevent overfitting.

## Evaluation

The trained model's performance was evaluated on a separate test dataset, which consisted of street view images from cities not seen during training. The evaluation metrics used include accuracy, precision, recall, and F1 score, providing insights into the model's ability to correctly predict the city in which an image was taken. Additionally, the model's performance was compared with human players in the GeoGuessr game to assess its competitiveness.
