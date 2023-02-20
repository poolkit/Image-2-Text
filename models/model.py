import tensorflow
from keras.layers import Input, Conv2D, Flatten, Dense, Embedding, LSTM, Dropout, Reshape, Concatenate, MaxPooling2D
from keras.models import Model

class BuildModel(tensorflow.keras.Model):
    def __init__(self, shape, max_length, vocab_length, output_dim, embed_matrix):
        super(BuildModel, self).__init__()

        self.image_input = Input(shape=shape)

        self.image_branch = Conv2D(8, (3, 3), strides=(1, 1), activation="relu")(self.image_input)
        self.image_branch = MaxPooling2D(pool_size=(2, 2),strides=2, padding='valid')(self.image_branch)

        self.image_branch = Conv2D(16, (3, 3), strides=(1, 1), activation="relu")(self.image_branch)
        self.image_branch = MaxPooling2D(pool_size=(2, 2),strides=2, padding='valid')(self.image_branch)

        self.flatten = Flatten()(self.image_branch)

        self.hidden_state = Dense(64, activation='tanh')(self.flatten)
        self.cell_state = Dense(64, activation='tanh')(self.flatten)

        self.text_input = Input(shape=(max_length,))
        self.embed = Embedding(input_dim=vocab_length, output_dim=output_dim, weights=[embed_matrix], trainable=False)
        self.text_embed = self.embed(self.text_input)

        self.lstm = LSTM(64, return_sequences=True, return_state=True)
        self.text_output, _, _ = self.lstm(self.text_embed, initial_state = [self.hidden_state, self.cell_state])

        self.dense = Dense(vocab_length, activation='softmax')
        self.final_output = self.dense(self.text_output)

        super().__init__([self.image_input, self.text_input], self.final_output)


