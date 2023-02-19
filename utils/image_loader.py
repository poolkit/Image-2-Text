import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import os


def image_generator(df, data_dir, batch_size):

    filepaths = [os.path.join(data_dir, fname) for fname in df.image]
    captions = [each for each in df.padded_one_hot]
    answers = [each for each in df.decoder_answers]


    # Loop over the file paths and load the images in batches
    while True:
        for i in range(0, len(filepaths), batch_size):
            batch_filepaths = filepaths[i:i+batch_size]
            batch_captions = captions[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            batch_images = []
            for j, filepath in enumerate(batch_filepaths):
                # Load the image and convert it to a NumPy array
                img = load_img(filepath, target_size=(224, 224))
                img_array = img_to_array(img)
                batch_images.append(img_array)
            yield (np.array(batch_images), np.array(batch_captions)), np.array(batch_answers)