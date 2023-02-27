from tqdm import tqdm
import os
import numpy as np
import tensorflow
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageProcessor():

    def __init__(self, transfer_model, image_shape, image_dir) -> None:
        self.transfer_model = transfer_model
        self.image_shape = image_shape
        if self.transfer_model == 'vgg16':
            self.image_model = VGG16(include_top=False, weights='imagenet',pooling='avg',input_shape=self.image_shape)
        elif self.transfer_model == 'resnet50':
            self.image_model = ResNet50(include_top=False, weights='imagenet',pooling='avg',input_shape=self.image_shape)
        self.image_feature_mapping = {}
        self.image_dir = image_dir

    def preprocess(self):
        for each in tqdm(os.listdir(self.image_dir)):
            img = load_img(f"{self.image_dir}/{each}", target_size=self.image_shape)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.image_model.predict(img_array)
            self.image_feature_mapping[each] = np.squeeze(features)
        
        return self.image_feature_mapping