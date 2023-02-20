import pandas as pd
import numpy as np
import os
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

class TextProcessor():

    def __init__(self, data_dir, caption_filename, max_length, output_dim) -> None:
        self.caption_file = os.path.join(data_dir,caption_filename)
        self.df = pd.read_csv(self.caption_file)
        self.max_length = max_length
        self.sentences = []
        self.vocab_dict = {}
        self.output_dim = output_dim
        self.word_to_id = {}
        self.id_to_word = {}
        self.max_ = 0

    def tokenize_with_tag(self,text):
        self.text = word_tokenize(text)
        self.text.insert(0,'<START>')
        self.text.append('<END>')
        return self.text

    def word2vec(self,sentences):
        self.w2v_model = Word2Vec(sentences=sentences,size=self.output_dim,window=10,min_count=1)
        self.w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
        return self.w2v_model

    def create_embed(self, w2v_model, op_dim, vocab, vocab_length):
        embed_dim = op_dim
        self.embed_matrix=np.zeros(shape=(vocab_length,embed_dim))
        for i,word in enumerate(vocab):
            if word == '<END>':
                self.embed_matrix[i]=np.zeros(embed_dim)
            else:
                embed_vector=w2v_model.wv.get_vector(word)
                self.embed_matrix[i]=embed_vector
        return self.embed_matrix

    def preprocess(self):
        self.df['tokenize'] = self.df['comment'].apply(self.tokenize_with_tag)
        self.df['word_count'] = self.df['tokenize'].str.len()
        self.df = self.df[self.df['word_count'] <= self.max_length]

        for each in self.df['tokenize']:
            self.sentences.append(each)

        self.w2v_model = self.word2vec(self.sentences)

        vocab_column = self.df['tokenize']
        for each in vocab_column:
            for i in each:
                if i in self.vocab_dict:
                    self.vocab_dict[i] += 1
                else:
                    self.vocab_dict[i] = 1

        print(f"Total number of words in vocab are {len(self.w2v_model.wv.vocab)}")

        self.vocab = list(self.vocab_dict.keys())
        self.vocab_length = len(self.vocab)

        self.embed_matrix = self.create_embed(self.w2v_model, self.output_dim, self.vocab, self.vocab_length)

        for i,each in enumerate(self.vocab):
            self.word_to_id[each] = i
            self.id_to_word[i] = each

        one_hot_sentences = []
        for each in self.sentences:
            temp = []
            for i in each:
                temp.append(self.word_to_id[i])
            if len(temp)>self.max_:
                self.max_ = len(temp)
            one_hot_sentences.append(temp)
        print(f"max_length: {self.max_}")

        padded_one_hot = pad_sequences(one_hot_sentences, maxlen=self.max_length, padding='post')
        decoder_answers = []
        for each in padded_one_hot:
            decoder_answers.append(each[1:])

        decoder_answers = pad_sequences(decoder_answers, maxlen=self.max_length, padding='post')

        self.df['padded_one_hot'] = padded_one_hot.tolist()
        self.df['decoder_answers'] = decoder_answers.tolist()

        self.final = self.df[['image','comment','padded_one_hot','decoder_answers']]

        return self.final