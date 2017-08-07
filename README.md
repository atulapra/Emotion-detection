# Emotion-detection

## Introduction

This project aims to find the emotion on faces with webcam using deep convolutional neural networks.

## Compatibility

* This code has been tested on Ubuntu 16.04 LTS and is most likely compatible on all platforms.

* **Dependencies**: Python 3.5+, [OpenCV 3.0](http://opencv.org/opencv-3-0.html), [TFlearn](http://tflearn.org/). 

## Usage

* Clone the repository and download the following files(trained models) from these links :
  * [data](https://drive.google.com/open?id=0B8_K9DW3E9PlV0phWlFfRGFfcEk)
  * [index](https://drive.google.com/open?id=0B8_K9DW3E9PlSmJySGM2Z0lwdlU)
  * [meta](https://drive.google.com/open?id=0B8_K9DW3E9Plb0ZVeHg0cEJuNlE)

* To run the program to detect emotions only in one face, type `python em_model.py singleface`.

* To run the program to detect emotions on all faces close to camera, type `python em_model.py multiface`.

## References

* [Emotion-recognition-and-prediction](https://github.com/nimish1512/Emotion-recognition-and-prediction)

* [Emotion-recognition-neural-networks](https://github.com/isseu/emotion-recognition-neural-networks)

