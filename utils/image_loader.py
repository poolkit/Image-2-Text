import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os

class ImageLoader():
    def __init__(self, df, data_dir, batch_size) -> None:
        self.df = df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.filepaths = [os.path.join(self.data_dir, fname) for fname in self.df.image]

    def image_generator(self):

        captions = [each for each in self.df.padded_one_hot]
        answers = [each for each in self.df.decoder_answers]

        # Loop over the file paths and load the images in batches
        while True:
            for i in range(0, len(self.filepaths), self.batch_size):
                batch_filepaths = self.filepaths[i:i+self.batch_size]
                self.batch_captions = captions[i:i+self.batch_size]
                self.batch_answers = answers[i:i+self.batch_size]
                self.batch_images = []
                for j, filepath in enumerate(batch_filepaths):
                    # Load the image and convert it to a NumPy array
                    img = load_img(filepath, target_size=(224, 224))
                    img_array = img_to_array(img)
                    self.batch_images.append(img_array)
                yield (np.array(self.batch_images), np.array(self.batch_captions)), np.array(self.batch_answers)