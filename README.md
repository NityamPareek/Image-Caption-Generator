# Image-Caption-Generator

# Objective

In this project, I aim to generate captions which describe a given input image, ie generate a 'caption' for the image. 

# Methodology

1. Used the flickr8k dataset which contains 8091 images, each having 5 captions.
2. Used an Encoder-Decoder based model with Pre-Trained VGG16 as Encoder and LSTM cell as Decoder.
3. Obtained Corpus Bleu-1 score of **0.539** and Corpus Bleu-2 score of **0.313** on the test data.

# User Guide

'''python
import nltk
'''
