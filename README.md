## Image Captioning
Image Captioning is the process of generating textual descriptions of an image. It is a challenging task in the field of computer vision and natural language processing. The goal of Image Captioning is to create a model that can generate a natural language description of an image, which should be accurate, relevant, and semantically meaningful.

This repository contains the code for building an Image Captioning model using Deep Learning. The model is trained on the Flickr30k dataset which contains a large collection of images with corresponding textual descriptions. The model is built using a Convolutional Neural Network (CNN) to extract features from the images and a Recurrent Neural Network (RNN) to generate the textual descriptions.

### Code Explaination
- ``text_processor.py`` : This script takes captions csv file and processes the captions as required by the model.
- ``image_processor.py`` : It creates a dictionary of image filenames mapped with their respective image feature vectors extracted from pre trained vgg16 or resnet50.
- ``model.py`` : The architecture of model. The model initially takes image input and extracts the features, which is later fed to LSTM layer along with processed captions.
- ``batch_loader.py`` : It loads all the image data along with caption with their respective targets in batches and yeilds a generator output.
- ``train.py`` : The training of the model happens here. The input is given from another script ``config.py``.
- ``inference.py`` : The caption generator code is here. Follow the below steps to generate captions for your own image

### Getting Started
1. Clone the repository

```sh
git clone https://github.com/poolkit/Image-2-Text.git
```

2. Download the [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) dataset. Unzip it and add it into ``'data/'`` folder.

3. Install all the required dependencies

```sh
pip install requirements.txt
```

4. To train the model, just run. It will save your model in ``saved/`` folder.

```sh
python train.py
```

5. Finnaly, to generate captions for your test images, rum

```sh
python inference.py  --t "test image filepath"  --o "output folder filepath" --m "saved model path"
```

### Results
![](results/image1.jpg)
![](results/image5.jpg)
![](results/image4.jpg)
![](results/image6.jpg)
![](results/image2.jpg)

### Future Work
Will add evaluation code for BLEU score and try Beam Search instead of Greedy Search
