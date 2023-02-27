import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class BatchLoader():
    def __init__(self, df, image_feature_map, batch_size, max_length, vocab_length) -> None:
        self.df = df
        self.image_feature_map = image_feature_map
        self.batch_size = batch_size
        self.filenames = [fname for fname in self.df.image]
        self.captions = [each for each in self.df.true_caption]
        self.max_length = max_length
        self.vocab_length = vocab_length

    def train_test_split(self):
        self.train_images, self.train_captions = [], []
        self.val_images, self.val_captions = [], []

        for i in range(np.shape(self.filenames)[0]):
            if i%8==0:
                self.val_images.append(self.filenames[i])
                self.val_captions.append(self.captions[i])
            else:
                self.train_images.append(self.filenames[i])
                self.train_captions.append(self.captions[i])

    def batch_generator(self, gen):

        file_names = self.train_images
        captions = self.train_captions
        length = len(self.train_images)

        if gen == 'val':
            file_names = self.val_images
            captions = self.val_captions
            length = len(self.val_images)

        # Loop over the file paths and load the images in batches
        while True:
            for i in range(0, length, self.batch_size):
                batch_filenames = file_names[i:i+self.batch_size]
                batch_captions = captions[i:i+self.batch_size]
                x_image, x_caption, y = [], [], []

                for filename, caption in zip(batch_filenames, batch_captions):
                    for j in range(1,self.max_length):
                        current_sequence = caption[0:j]
                        target_word = caption[j]
                        current_sequence = pad_sequences([current_sequence], maxlen=self.max_length, padding='post')
                        one_hot_target_word = to_categorical([target_word], self.vocab_length)[0]
                        x_image.append(self.image_feature_map[filename])
                        x_caption.append(current_sequence)
                        y.append(one_hot_target_word)
                yield (np.array(x_image), np.array(x_caption).squeeze()), np.array(y)