from models.model import BuildModel
from utils.image_loader import ImageLoader
from utils.text_processor import TextProcessor
from utils.config import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import tensorflow as tf
# tf.config.run_functions_eagerly(True)


class Trainer():

    def __init__(self, data_dir, caption_filename, batch_size, image_shape, max_length, output_dim, epochs) -> None:
        self.data_dir = data_dir
        self.caption_filename = caption_filename
        self.batch_size = batch_size
        self.shape = image_shape
        self.max_length = max_length
        self.output_dim = output_dim
        self.epochs = epochs

    def save_checkpoints(self):
        try:
            os.mkdir('saved_weights')
        except:
            pass 
        # Define the file path for saving the model weights
        self.checkpoint_filepath = 'saved_weights/model.{epoch:02d}-{acc:.2f}.h5'
        # Create a ModelCheckpoint callback that saves the best model weights based on validation accuracy
        self.checkpoint_callback = ModelCheckpoint(
            self.checkpoint_filepath,
            monitor='acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )
        # Create an EarlyStopping callback that stops training when validation accuracy reaches 0.95
        self.early_stopping_callback = EarlyStopping(
            monitor='acc',
            patience=5,
            verbose=1,
            mode='max',
            baseline=0.95
        )
        return [self.checkpoint_callback, self.early_stopping_callback]

    def train_model(self):

        tp = TextProcessor(self.data_dir, self.caption_filename, self.max_length, self.output_dim)
        df = tp.preprocess()

        il = ImageLoader(df, self.data_dir, self.batch_size)

        model = BuildModel(self.shape, self.max_length, tp.vocab_length, self.output_dim, tp.embed_matrix)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

        generator = il.image_generator()
        num_batches = len(il.filepaths)//self.batch_size
        print(f"num_batches: {num_batches}")

        history = model.fit(generator,steps_per_epoch=num_batches,epochs=self.epochs,callbacks=self.save_checkpoints())

        return history

if __name__ == '__main__':
    trainer = Trainer(DATA_DIR, CAPTION_FILENAME, BATCH_SIZE,IMAGE_SHAPE,MAX_LENGTH,OUTPUT_DIM,EPOCHS)
    hist = trainer.train_model()

