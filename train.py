from models.model import BuildModel
from utils.image_processor import ImageProcessor
from utils.text_processor import TextProcessor
from utils.batch_loader import BatchLoader
from utils.config import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pickle
import tensorflow
tensorflow.config.run_functions_eagerly(True)


class Trainer():

    def __init__(self, image_dir, caption_filename, batch_size, transfer_model, image_shape, max_length, output_dim, epochs) -> None:
        self.image_dir = image_dir
        self.caption_filename = caption_filename
        self.batch_size = batch_size
        self.transfer_model = transfer_model
        self.shape = image_shape
        self.max_length = max_length
        self.output_dim = output_dim
        self.epochs = epochs

    def save_checkpoints(self):
        # Define the file path for saving the model weights
        self.checkpoint_filepath = 'saved/weights/model.{epoch:02d}-{acc:.2f}.h5'
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

        text_process = TextProcessor(self.caption_filename, self.max_length, self.output_dim)
        df = text_process.preprocess()

        try:
            os.mkdir('saved/')
            os.mkdir('saved/weights/')
        except:
            pass 

        if f"image_feature_mapping_{self.transfer_model}.p" in os.listdir("saved"):
            with open(f"saved/image_feature_mapping_{self.transfer_model}.p", 'rb') as f:
                image_feature_mapping = pickle.load(f)
            print("Image feature mapping already found. Using that.")
        else:
            image_mapping = ImageProcessor(self.transfer_model, self.shape, self.image_dir)
            image_feature_mapping = image_mapping.preprocess()
            with open( f"saved/image_feature_mapping_{self.transfer_model}.p", "wb" ) as f:
                pickle.dump(image_feature_mapping, f )
        
        model = BuildModel(self.transfer_model, self.max_length, text_process.vocab_length, self.output_dim, text_process.embed_matrix)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        loading_batch = BatchLoader(df, image_feature_mapping, self.batch_size, self.max_length, text_process.vocab_length)
        loading_batch.train_test_split()
        train_generator = loading_batch.batch_generator(gen='train')
        val_generator = loading_batch.batch_generator(gen='val')
        train_steps = len(loading_batch.train_images)//self.batch_size
        val_steps = len(loading_batch.val_images)//self.batch_size
        print(f"train_steps: {train_steps}")
        print(f"val_steps: {val_steps}")

        print("**************************** Starting the training ****************************")
        history = model.fit(train_generator, steps_per_epoch=train_steps, 
                    epochs=self.epochs, validation_data=val_generator, validation_steps=val_steps, 
                    callbacks=self.save_checkpoints())

        return history

if __name__ == '__main__':
    trainer = Trainer(IMAGE_DIR, CAPTION_FILENAME, BATCH_SIZE, TRANSFER_MODEL, IMAGE_SHAPE, MAX_LENGTH, OUTPUT_DIM, EPOCHS)
    history = trainer.train_model()