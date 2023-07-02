# Image-Caption-Generator

# Objective

In this project, I aim to generate text which describe a given input image, ie generate a 'caption' for the image. 

# Methodology

1. Used the flickr8k dataset which contains 8091 images, each having 5 captions.
2. Used an Encoder-Decoder based model with Pre-Trained VGG16 as Encoder and LSTM cell as Decoder.
3. Obtained Corpus Bleu-1 score of **0.539** and Corpus Bleu-2 score of **0.313** on the test data.

# Model Architecture

![Model](/model.png)

# User Guide

Steps to run this on your local computer:

- Clone this repository
```
git clone https://github.com/NityamPareek/Image-Caption-Generator
```

- Make a new virtual environment in python in the folder in which this repository is saved and Active the Environment.
```
pip install virtualenv
python -m venv <myenvname> 
path\to\venv\Scripts\Activate.ps1  (Run this line with your path to activate the virtual environment)
```
- Download the requirements of the environment using 
```
pip install -r requirements.txt
```
- Download the model and place it into the same directory as app.py. You can download the model [here](https://drive.google.com/file/d/1Wjwzla4oB5OLOqO_G8BEwoOPxchH6JLG/view?usp=sharing).
- Run app.py file to open the website
```
streamlit run app.py
```

# Website Screenshots
- This is the homepage you will see in the browser. To upload your image, click on browse and select an image from your computer.
![Homepage](/homepage.png)
- The results will appear below your image under the 'Generated Caption' heading
![Output](/output.png)
