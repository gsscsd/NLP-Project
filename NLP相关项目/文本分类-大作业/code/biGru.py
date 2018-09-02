import pandas as pd 
import numpy as np
np.random.seed(2018)
import re
import jieba
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
import keras
from gensim.models import word2vec
import gensim
import logging
from utils import *
import warnings

warnings.filterwarnings('ignore')



###w2v的特征维度
n_dim = 300
max_features = 30000
maxlen = 50
embed_size = 300

##中文分词




###对词向量的转换
def gen_embed_matrix(word_index,w2v_model):
    nb_words = min(max_features, len(word_index))
    ###计算权重
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: 
            continue
        else :
            try:
                embedding_vector = w2v_model[word]
            except KeyError:
                continue
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


###keras的model函数式，copy from  kaggle
def get_model(max_features, embed_size, embedding_matrix):
	inp = Input(shape=(maxlen, ))
	x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
	x = SpatialDropout1D(0.2)(x)
	x = Bidirectional(GRU(256, return_sequences=True))(x)
	x = Conv1D(128,kernel_size = 2,padding = 'valid',kernel_initializer = 'he_uniform')(x)
	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)
	conc = concatenate([avg_pool, max_pool])
	outp = Dense(1, activation="sigmoid")(conc)

	###这里的参数可以修正
	model = Model(inputs=inp, outputs=outp)
	model.compile(loss='binary_crossentropy',
	              optimizer='sgd',
	              metrics=['accuracy'])

	return model



def main():
    print("*+" *30)
    df = pd.read_pickle("../temp/df_train.pkl")
    y = df['y'].values
    save_model_name = "../temp/word2vec.vec"
    w2v_model = word2vec.Word2Vec.load(save_model_name)
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(df['cut_words']))
    train = tokenizer.texts_to_sequences(df['cut_words'])
    train = sequence.pad_sequences(train, maxlen=maxlen)
    word_index = tokenizer.word_index
    embedding_matrix = gen_embed_matrix(word_index,w2v_model)
    print("starting ....")
    model = get_model(max_features, embed_size, embedding_matrix)
    print(model.summary())
    X_tra, X_val, y_tra, y_val = train_test_split(train, y, train_size=0.85, random_state=233)
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)

    #创建一个实例history
    history = LossHistory()

    batch_size = 128
    epochs = 50
    hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = [early_stopping,history])
    print("all is Ok.")

    #绘制acc-loss曲线
    history.loss_plot('epoch')
    print("*+" *30)
    pass

if __name__ == '__main__':
    main()
