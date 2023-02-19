import pandas as pd
import numpy as np
from config import *
import os
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import gensim
from keras.preprocessing.sequence import pad_sequences

def clean_text(text):
  text = text.lower()
  text = re.sub(r"\'m", " am", text)
  text = re.sub(r"\'s", " is", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r" 'bout", " about", text)
  text = re.sub(r"gonna", "going to", text)
  text = re.sub(r"gotta", "got to", text)
  text = re.sub(r"won't", "will not", text)
  text = re.sub(r"can't", "can not", text)
  text = re.sub(r"n't", " not", text)
  text = re.sub(r"-"," ",text)
  return text

def tokenize_with_tag(text):
  text = word_tokenize(text)
  text.insert(0,'<START>')
  text.append('<END>')
  return text

def word2vec(sentences):
    w2v_model = gensim.models.Word2Vec(sentences=sentences,size=output_dim,window=10,min_count=1)
    w2v_model = w2v_model.train(sentences,epochs=10,total_examples=len(sentences))
    return w2v_model

# create embedding matrix
def create_embed(w2v_model, op_dim, vocab, vocab_length):
    embed_dim = op_dim
    embed_matrix=np.zeros(shape=(vocab_length,embed_dim))
    for i,word in enumerate(vocab):
        if word == '<END>':
            embed_matrix[i]=np.zeros(embed_dim)
        else:
            embed_vector=w2v_model.get_vector(word)
            embed_matrix[i]=embed_vector
    return embed_matrix

def preprocess():
    caption_file = os.path.join(data_dir,'results.csv')
    df = pd.read_csv(caption_file)
    
    df['comment'] = df['comment'].apply(clean_text)
    df['tokenize'] = df['comment'].apply(tokenize_with_tag)
    df['word_count'] = df['tokenize'].str.len()
    df = df[df['word_count'] <= max_length]

    sentences = []
    for each in df['tokenize']:
        sentences.append(each)

    w2v_model = word2vec(sentences)
    
    vocab_column = df['tokenize']
    vocab_dict = {}

    for each in vocab_column:
        for i in each:
            if i in vocab_dict:
                vocab_dict[i] += 1
            else:
                vocab_dict[i] = 1

    print(f"Total number of words in vocab are {len(w2v_model.vocab)}")

    vocab = list(vocab_dict.keys())
    vocab_length = len(vocab)

    embed_matrix = create_embed(w2v_model, output_dim, vocab, vocab_length)

    word_to_id = {}
    id_to_word = {}
    for i,each in enumerate(vocab):
        word_to_id[each] = i
        id_to_word[i] = each

    one_hot_sentences = []
    max_ = 0
    for each in sentences:
        temp = []
    for i in each:
        temp.append(word_to_id[i])
    if len(temp)>max_:
        max_ = len(temp)
    one_hot_sentences.append(temp)
    print(f"max_length: {max_}")
    one_hot_sentences[:5]
    
    padded_one_hot = pad_sequences(one_hot_sentences, maxlen=max_length, padding='post')

    decoder_answers = []

    for each in padded_one_hot:
        decoder_answers.append(each[1:])

    decoder_answers = pad_sequences(decoder_answers, maxlen=max_length, padding='post')

    df['padded_one_hot'] = list(padded_one_hot)
    df['decoder_answers'] = list(decoder_answers)

    final = df[['image_name','comment','padded_one_hot','decoder_answers']]

    return final, embed_matrix

x, y = preprocess()
print(x)                  

