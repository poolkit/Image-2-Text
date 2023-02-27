import tensorflow
from keras.layers import Input, Conv2D, Flatten, Dense, Embedding, LSTM, Dropout, Reshape, Concatenate, MaxPooling2D, Add
from keras.models import Model

class BuildModel():
    def __init__(self, transfer_model, max_length, vocab_length, output_dim, embed_matrix):
        self.transfer_model = transfer_model
        self.max_length = max_length
        self.vocab_length = vocab_length
        self.output_dim = output_dim
        self.embed_matrix = embed_matrix

    def create_model(self):

        if self.transfer_model == 'vgg16':
            self.shape = (512,)
        elif self.transfer_model == 'resnet50':
            self.shape = (2048,)

        self.image_input = Input(shape=self.shape, name='image_input')
        self.image_output = Dense(256, activation='relu', name='image_dense')(self.image_input)

        self.text_input = Input(shape=(self.max_length,), name='text_input')
        self.embed = Embedding(input_dim=self.vocab_length, output_dim=self.output_dim, weights=[self.embed_matrix], trainable=False, name='embedding')
        self.text_embed = self.embed(self.text_input)

        self.lstm = LSTM(256, name='lstm')
        self.text_output = self.lstm(self.text_embed)

        self.combine = Add()([self.image_output, self.text_output])
        self.final_dense = Dense(256,activation='relu', name='combine_dense')(self.combine)
        self.final_output = Dense(self.vocab_length, activation='softmax', name='final_dense')(self.final_dense)

        self.model = Model([self.image_input, self.text_input], self.final_output)
        print(self.model.summary())
        
        return self.model


