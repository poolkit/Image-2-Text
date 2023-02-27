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

    def __init__(self, caption_filename, max_length, output_dim) -> None:
        self.caption_file = caption_filename
        self.df = pd.read_csv(self.caption_file)
        self.max_length = max_length
        self.sentences = []
        self.vocab_dict = {}
        self.output_dim = output_dim
        self.word_to_id = {}
        self.id_to_word = {}

    def tokenize_with_tag(self,text):
        self.text = text.lower()
        self.text = word_tokenize(text)
        self.text.insert(0,'<start>')
        self.text.append('<end>')
        return self.text

    def word2vec(self,sentences):
        self.w2v_model = Word2Vec(sentences=sentences,size=self.output_dim,window=10,min_count=1)
        self.w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
        return self.w2v_model

    def create_embed(self, w2v_model, op_dim, vocab, vocab_length):
        embed_dim = op_dim
        self.embed_matrix=np.zeros(shape=(vocab_length,embed_dim))
        for i,word in enumerate(vocab):
            if word == '<out>':
                self.embed_matrix[i]=np.zeros(embed_dim)
            else:
                embed_vector=w2v_model.wv.get_vector(word)
                self.embed_matrix[i]=embed_vector
        return self.embed_matrix

    def preprocess(self):
        self.df['tokenize'] = self.df['caption'].apply(self.tokenize_with_tag)
        self.df['word_count'] = self.df['tokenize'].str.len()

        min_length = 8
        self.df = self.df[self.df['word_count'] <= self.max_length]
        self.df = self.df[self.df['word_count'] >= min_length]

        count_column = self.df['tokenize']
        count_dict = {}
        for each in count_column:
            for i in each:
                if i in count_dict:
                    count_dict[i] += 1
                else:
                    count_dict[i] = 1

        threshold = 5

        self.vocab_dict = {key: value for key, value in count_dict.items() if value > threshold}

        self.sentences = []
        
        for each in self.df['tokenize']:
            temp = []
            for i in each:
                if i in self.vocab_dict:
                    temp.append(i)
                else:
                    temp.append('<out>')
            self.sentences.append(temp)
        self.w2v_model = self.word2vec(self.sentences)

        print(f"Total number of words in vocab are {len(self.w2v_model.wv.vocab)}")

        self.vocab = {k for k,v in self.w2v_model.wv.vocab.items()}
        self.vocab_length = len(self.w2v_model.wv.vocab)

        self.embed_matrix = self.create_embed(self.w2v_model, self.output_dim, self.vocab, self.vocab_length)

        for i,each in enumerate(self.vocab):
            self.word_to_id[each] = i
            self.id_to_word[i] = each

        one_hot_sentences = []
        max_ = 0
        for each in self.sentences:
            temp = []
            for i in each:
                temp.append(self.word_to_id[i])
            if len(temp)>max_:
                max_ = len(temp)
            one_hot_sentences.append(temp)
        print(f"max_length: {max_}")
        
        true_caption = pad_sequences(one_hot_sentences, maxlen=self.max_length, padding='post')

        self.df['true_caption'] = true_caption.tolist()

        self.df = self.df[['image','caption','true_caption']]
        self.df = self.df.sample(frac=1).reset_index(drop=True)

        return self.df