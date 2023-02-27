from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
from models.model import BuildModel
from utils.config import *
from utils.image_processor import ImageProcessor
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from PIL import Image


loaded_model = load_model("saved/model_23-57.h5")

with open('saved/objects.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    word_to_id, id_to_word = pickle.load(f)

def generate_captions(filepath):

    image_class = ImageProcessor(TRANSFER_MODEL, IMAGE_SHAPE, TEST_DIR)
    image_model = image_class.image_model

    image = load_img(filepath, target_size=(224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = image_model.predict(img_array)

    caption = '<start>'

    for i in range(MAX_LENGTH):
        sequence = [word_to_id[i] if i in word_to_id else word_to_id['<out>'] for i in caption.split(' ')]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding='post')
        pred = loaded_model.predict([img_array, sequence], verbose=0)
        pred = np.argmax(pred[0])
        word = id_to_word[pred]
        
        if word != '<end>':
            caption += ' ' + word
        else:
            break
    return caption[8:]

if __name__ == '__main__':

    test_image_path = "data/test/image6.jpg"
    generated_caption  = generate_captions(test_image_path)
    test_image = Image.open(test_image_path)
    plt.imshow(test_image)
    plt.title(generated_caption)
    plt.show()