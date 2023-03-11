## Image Captioning
Image Captioning is the process of generating textual descriptions of an image. It is a challenging task in the field of computer vision and natural language processing. The goal of Image Captioning is to create a model that can generate a natural language description of an image, which should be accurate, relevant, and semantically meaningful.

This repository contains the code for building an Image Captioning model using Deep Learning. The model is trained on the Flickr30k dataset which contains a large collection of images with corresponding textual descriptions. The model is built using a Convolutional Neural Network (CNN) to extract features from the images and a Recurrent Neural Network (RNN) to generate the textual descriptions.

### Code Explanation
- ``text_processor.py`` : This script takes captions csv file and processes the captions as required by the model.
- ``image_processor.py`` : It creates a dictionary of image filenames mapped with their respective image feature vectors extracted from pre trained vgg16 or resnet50.
- ``model.py`` : The architecture of model. The model initially takes image input and extracts the features, which is later fed to LSTM layer along with processed captions.
- ``batch_loader.py`` : It loads all the image data along with caption with their respective targets in batches and yeilds a generator output.
- ``train.py`` : The training of the model happens here. The input is given from another script ``config.py``.
- ``inference.py`` : The caption generator code is here. Follow the below steps to generate captions for your own image.
- ``app.py`` : This is a FastAPI endpoint script that allows you to provide a url to generate captions.

### Getting Started
1. Clone the repository

```sh
git clone https://github.com/poolkit/Image-2-Text.git
```

2. Download the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset. Unzip it and add it into ``'data/'`` folder.

3. Install all the required dependencies

```sh
pip install requirements.txt
```

4. To train the model, just run. It will save your model in ``saved/`` folder. (resnet50 gave better accuracy btw)

```sh
python train.py
```

5. Finally, to generate captions for your test images, run
```sh
python inference.py  --t "test image filepath"
```
or
```sh
python inference.py  --t "test image filepath"  --o "output folder filepath" --m "saved model path"
```

6. To get results from an API endpoint, run
```sh
python app.py
```
All you need to do is paste the url of the image you wanna test.

### Results
<img src="results/image1.jpg" width=500px height=300px> <img src="results/image5.jpg" width=500px height=300px>
<img src="results/image4.jpg" width=500px height=300px> <img src="results/image6.jpg" width=500px height=300px>
<img src="results/image2.jpg" width=500px height=300px>

### Future Work
Will add evaluation code for BLEU score and try Beam Search instead of Greedy Search
