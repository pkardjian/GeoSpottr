# GeoSpottr: Image-Based Geolocalization

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)


## Introduction

GeoGuessr is an online game where players are shown a random street view image from anywhere in the world and have to guess the location. This project aims to develop a model that can predict the city in which an image was taken by analyzing visual features extracted from the street view images. By utilizing the power of Convolutional Neural Networks (CNN) and the ResNet50 architecture, we can train a highly accurate image classifier.

![alt text](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjuVSRtsMiLntaMRx9oAiPgsPetFItAc9MsC7hBIVD5p6PJDA6UJxrgZGQXMVx7KhhfXcqP-NTgwlHLuuJWy1yJ_eAjgn-12FmYNhKvSUwaAKyj1oGfv-pyxMg8M7NsWt7TzqSdsCOZ59lEG-ej-BuZr4j4NSfJctGCHcdPmzD5poYNs2_aQGxySra3Ww/s1822/1.png)

### Methods Used
- Data Filtering
- Image Transformation
- Network Architecture Research
- Convolutional Neural Networks
- Model Training

### Technologies
- Python
- Google Street View API
- PyTorch, Torchvision, NumPy, PIL
- Google Colaboratory, Visual Studio Code

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

The implementation and training of the model was done in Python, most notably using the PyTorch and Torchvision libraries. 

The training process involved fine-tuning the model's weights using Cross-Entropy loss and Stochastic Gradient Descent (SGD) optimizer. Training, testing and validation sets were created using a 60-20-20 split to help monitor the model's performance and prevent overfitting. Extensive hyperparameter tuning was conducted in order to ensure optimal performance.

## Testing

One of our main objectives in testing our model against new data was to emulate a human playing the GeoGuessr game. On the GeoGuessr platform, users can create custom maps using coordinates which GeoGuessr then uses to extract panoramas to be used for the game. As such, we created a map containing images from our test set that can then be played by a human. Their performance can thenbe compared to the test accuracy of our model. In order for accurate comparison, players must disable any moving, zooming or panning as the input to the model is a static image rather than the dynamic panoramas that GeoGuessr provides. 

Further improvements to this testing method can include taking a screenshot of the GeoGuessr panorama and directly sending it to the model for prediction via a back-end server. 

## Results

Our best model had a final training accuracy of 65.3% and a final validation accuracy of 57.6%. In addition to this, our final model training loss was recorded at slightly above 0.03. 

![alt text](https://github.com/pkardjian/GeoSpottr/blob/main/primary/results/TrainingCurves.jpg)

When our model was evaluated against the test set, the test accuracy was determined to be 61.9% with both a weighted precision and recall score of 0.62. From the confusion matrix, we also observe that the North American cities (Toronto, Chicago, New York and Los Angeles) were often falsely predicted for each other. We also see that the city classes with the worst and best accuracy were Melbourne and Berlin who reported accuracies of 37.1% and 80.3%, respectively. 

![alt text](https://github.com/pkardjian/GeoSpottr/blob/main/primary/results/ConfusionMatrix.png)


