from models.model import BuildModel
from utils import image_loader
from utils import text_preprocesser as tp
from utils.config import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

os.mkdir('saved_weights') 
# Define the file path for saving the model weights
checkpoint_filepath = 'saved_weights/model.{epoch:02d}-{acc:.2f}.h5'
# Create a ModelCheckpoint callback that saves the best model weights based on validation accuracy
checkpoint_callback = ModelCheckpoint(
    checkpoint_filepath,
    monitor='acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='max'
)
# Create an EarlyStopping callback that stops training when validation accuracy reaches 0.95
early_stopping_callback = EarlyStopping(
    monitor='acc',
    patience=5,
    verbose=1,
    mode='max',
    baseline=0.95
)

def train_model():

    df, embed_matrix, vocab_length = tp.preprocess()
    num_batches = df.shape[0]//batch_size

    model = BuildModel(shape, max_length, vocab_length, output_dim, embed_matrix)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    generator = image_loader.image_generator(df, data_dir, batch_size)

    history = model.fit(generator,steps_per_epoch=num_batches,epochs=epochs,callbacks=[checkpoint_callback, early_stopping_callback])

    return history

if __name__ == '__main__':
    history = train_model()


