# Emotion-detection

## Introduction

This project aims to classify the emotion on a person's face into one of **seven categories**, using deep convolutional neural networks. This repository is an implementation of [this](https://github.com/atulapra/Emotion-detection/blob/master/ResearchPaper.pdf) research paper.

## Dependencies

* Python 3.5+, [OpenCV 3.0](http://opencv.org/opencv-3-0.html), [TFlearn](http://tflearn.org/).

## Usage

* Clone the repository and download the **trained model** files from [here](https://drive.google.com/file/d/1rdgSdMcXIvfoPmf702UCtH6RNcvkKFu7/view?usp=sharing), extract it and copy the files into the current working directory.

* To run the program to detect emotions only in **one face**, type `python em_model.py singleface`.

* To run the program to detect emotions on all faces close to camera, type `python em_model.py multiface`. Note that this sometimes generates incorrect predictions.

* The folder structure is of the form:

  Emotion-detection:
  * emojis (folder)
  * examples (folder)
  * haarcascade_files (folder)
  * `em_model.py` (file)
  * `multiface.py` (file)
  * `singleface.py` (file)
  * `README.md` (file)
  * `model_1_atul.tflearn.data-00000-of-00001` (file)
  * `model_1_atul.tflearn.index` (file)
  * `model_1_atul.tflearn.meta` (file)

## Algorithm

* First, we use **haar cascade** to detect faces in each frame of the webcam feed.

* The region of image containing the face is resized to **48x48** and is passed as input to the ConvNet.

* The network outputs a list of **softmax scores** for the seven classes.

* The emotion with maximum score is displayed on the screen.

## Example Output

![Happy](examples/happy.png)
