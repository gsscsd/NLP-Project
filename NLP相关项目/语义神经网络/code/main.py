"""
程序的整个流程
1.读入训练集，验证集，测试集和新闻body集
2.合并新闻body到训练集，验证集和测试集
3.对于所有的数据集，需要清洗headline和body，分词，去除停用词
5.使用keras内置的函数，把headline文本和body文本转换成序列数字，并且对齐数据headline 100，body 300，这两个参数可以调节
6.读入word2vec，构建词表词向量
7.创建lstm——DSSM模型，并且输出模型的结构图
8.训练数据（4-6）个小时，打印训练过程的acc,val_acc,loss,val_loss
9.输出验证集的评测结果
10.保存测试集的结果
"""

import pandas as pd 
import numpy as np 
import nltk
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
# import matplotlib.pyplot as plt 
# import seaborn as sns
# sns.set( palette="muted", color_codes=True)



# 文件的路径
dir_train = "../data/train_data.csv"
dir_val = "../data/validation_data.csv"
dir_test = "../data/test_data.csv"
dir_body = "../data/article_body_texts.csv"
dir_word2vec = r"F:\Project_Jupyter\google_word2vec\GoogleNews-vectors-negative300.bin.gz"

# 文本的长度参数设置
MAX_SEQUENCE_LENGTH_0 = 100
MAX_SEQUENCE_LENGTH_1 = 300
# 预定义词表的大小
MAX_NB_WORDS = 200000
# 词向量的维度
EMBEDDING_DIM = 300
# 训练集切分的比例
VALIDATION_SPLIT = 0.1

# 参数设置，随机数，lstm单元，隐藏层单元的数目
# lstm单元的数目，用np随机产生，（175，275）之间
num_lstm = np.random.randint(175, 275)
# 隐藏层单元的数目，用np随机产生，（100，150）之间
num_dense = np.random.randint(100, 150)
# drop层的 参数
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
# 非线性参数，relu,tanh
act = 'relu'
# 批量进行运算的数据的大小
batch_size = 128
# 迭代运行的次数
epochs = 100


# 读入数据
df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)
df_val = pd.read_csv(dir_val)
df_body = pd.read_csv(dir_body)

# 把新闻的body和headline merge到一起
df_train_temp = pd.merge(df_train,df_body,on = 'Body ID',how = 'left')
df_val_temp = pd.merge(df_val,df_body,on = 'Body ID',how = 'left')
df_test_temp = pd.merge(df_test,df_body,on = 'Body ID',how = 'left')

# 加载预处理的词向量
word2vec = KeyedVectors.load_word2vec_format(datapath(dir_word2vec),
                                                binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))

# 分词去除停用词
print('Processing text dataset')
def text_to_wordlist(text):
    # 清洗文本数据
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\s{2,}", " ", text)
    # 去除乱七八糟的标点和符号
    text = re.sub(r"[!@#$%^&*()_+-=,./:;{}|]",' ',text)
    # 去除数字
    text = re.sub(r"[0-9]"," ",text)
    
    text = text.lower().split()
    # 去除停用词
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    # 去除空格
    text = [str(i) for i in text if len(i) > 1]
    
    text = " ".join(text)

    return(text)


# 清理文本，分词，停用词
df_train_temp['Headline'] = df_train_temp['Headline'].map(text_to_wordlist)
df_val_temp['Headline'] = df_val_temp['Headline'].map(text_to_wordlist)
df_test_temp['Headline'] = df_test_temp['Headline'].map(text_to_wordlist)
df_train_temp['articleBody'] = df_train_temp['articleBody'].map(text_to_wordlist)
df_val_temp['articleBody'] = df_val_temp['articleBody'].map(text_to_wordlist)
df_test_temp['articleBody'] = df_test_temp['articleBody'].map(text_to_wordlist)

# 把标签列转换成数字表示，并且用来去做评测
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]
# 这个是从文本到数字转换的map
label_map = {j:i for i,j in enumerate(LABELS)}
# 这个是从数字到文本转换的map
num2label_map = {i:j for i,j in enumerate(LABELS)}

# label列转换成onehot
df_train_temp["Stance"] = df_train_temp["Stance"].map(label_map)
df_val_temp["Stance"] = df_val_temp["Stance"].map(label_map)
y_train = df_train_temp["Stance"]
y_val = df_val_temp["Stance"]
# 转换成one-hot
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# 首先把所有的数据类型转换成str类型，因为其类型是object类型，所以需要转换
print("precosing types: ")
df_train_temp['Headline'] = df_train_temp['Headline'].astype('str')
df_val_temp['Headline'] = df_val_temp['Headline'].astype('str')
df_test_temp['Headline'] = df_test_temp['Headline'].astype('str')
df_train_temp['articleBody'] = df_train_temp['articleBody'].astype('str')
df_val_temp['articleBody'] = df_val_temp['articleBody'].astype('str')
df_test_temp['articleBody'] = df_test_temp['articleBody'].astype('str')

# 文本转换成序列数字表示
# 类似于['i','o'] = [1,2]
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df_body['articleBody'])

# 转换headline的文本成序列数字
train_head = tokenizer.texts_to_sequences(df_train_temp['Headline'])
val_head = tokenizer.texts_to_sequences(df_val_temp['Headline'])
test_head = tokenizer.texts_to_sequences(df_test_temp['Headline'])

# 转换body的文本成序列数字
train_body = tokenizer.texts_to_sequences(df_train_temp['articleBody'])
val_body = tokenizer.texts_to_sequences(df_val_temp['articleBody'])
test_body = tokenizer.texts_to_sequences(df_test_temp['articleBody'])

# 计算一共有多少个词，从文本转换成数字
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# 生成词向量矩阵
print('Preparing embedding matrix')
# 首先生成一个词表向量矩阵[nb_words,300],也就是有nb_words个词，每个词有300维
nb_words = min(MAX_NB_WORDS, len(word_index))+1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# 对齐文本，head和body分别对齐到固定的长度，head为100，body是300
print("word to sequence...")
# train的headline数据
train_head_data = sequence.pad_sequences(train_head, maxlen=MAX_SEQUENCE_LENGTH_0)
# val的headline数据
val_head_data = sequence.pad_sequences(val_head, maxlen=MAX_SEQUENCE_LENGTH_0)
# test的headline数据
test_head_data = sequence.pad_sequences(test_head, maxlen=MAX_SEQUENCE_LENGTH_0)
# train的body数据
train_body_data = sequence.pad_sequences(train_body, maxlen=MAX_SEQUENCE_LENGTH_1)
# val的body数据
val_body_data = sequence.pad_sequences(val_body, maxlen=MAX_SEQUENCE_LENGTH_1)
# test的body数据
test_body_data = sequence.pad_sequences(test_body, maxlen=MAX_SEQUENCE_LENGTH_1)

# 创建一个模型
def create_model():
    ########################################
    ## define the model structure
    ########################################
    # 第一个词向量层，用于表示headline
    embedding_layer_0 = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH_0,
            trainable=False)
    # 第二个词向量层，用于表示body
    embedding_layer_1 = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH_1,
            trainable=False)
    # lstm层
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    
    # 第一个层的参数设置
    sequence_0_input = Input(shape=(MAX_SEQUENCE_LENGTH_0,), dtype='int32')
    embedded_sequences_0 = embedding_layer_0(sequence_0_input)
    x1 = lstm_layer(embedded_sequences_0)

    # 第二个层的参数设置
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH_1,), dtype='int32')
    embedded_sequences_1 = embedding_layer_1(sequence_1_input)
    y1 = lstm_layer(embedded_sequences_1)

    # 连接两个层
    merged = concatenate([x1, y1])
    # drop 一下
    merged = Dropout(rate_drop_dense)(merged)
    # bathcnamorlization，归一化处理
    merged = BatchNormalization()(merged)
    # 隐藏层
    merged = Dense(num_dense, activation=act)(merged)
    # drop 一下
    merged = Dropout(rate_drop_dense)(merged)
    # batch一下
    merged = BatchNormalization()(merged)
    # 最后一个输出层，4分类
    preds = Dense(4, activation='softmax')(merged)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_0_input, sequence_1_input], \
            outputs=preds)
    model.compile(loss='categorical_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    return model


# 生成模型
model = create_model()
# 输出模型的结构
print(model.summary())
# 保存模型的结构图
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model1.png',show_shapes=True)
print("Save model.")


# 保存最优模型的参数
STAMP = "../temp/" + 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)
# 训练过程中，如果val_loss三轮之后，都比前一轮要高，那么，停止训练
early_stopping =EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
print(STAMP)

# 训练模型，大约需要4-6个小时，
hist = model.fit([train_head_data, train_body_data], y_train,
        # 验证集的切分
        validation_split = 0.15,
        # 迭代的次数，batch的数目，是否打乱
        epochs=3, batch_size=batch_size, shuffle=True, 
        callbacks=[early_stopping, model_checkpoint])

# plot训练过程中，精度和损失
# 画出训练过程中的精度和损失
# def plot_precoss(hist):
#     plt.figure(figsize=(12,8))
#     plt.plot(hist.history['acc'])
#     plt.plot(hist.history['val_acc'])
#     plt.plot(hist.history['loss'])
#     plt.plot(hist.history['val_loss'])
#     plt.title("acc and loss")
#     plt.xlabel("number")
#     plt.ylabel("precesion")
#     plt.legend(['acc','val_acc','loss','val_loss'],loc = 7,fontsize = 'large')
#     plt.show()



# 验证验证集上的评分是多少，先进行预测
y_val_temp = model.predict([val_head_data, val_body_data])
# 然后，对于预测的label是onehot，然后转换成0-3之间
y_val_temp_p = np.argmax(y_val_temp,axis = 1)
y_val_temp_t = np.argmax(y_val,axis = 1)
# 0-3之间的数字，在转换成文本表示，agree，disagree等等。
y_val_temp_p = [LABELS[i] for i in y_val_temp_p]
y_val_temp_t = [LABELS[i] for i in y_val_temp_t]


# 测评函数
def score_submission(gold_labels, test_labels):
    # score是评分，cm是混淆矩阵
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]
    # 根据官网给定的公式来计算评分
    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25
        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1
    return score, cm

# 输出混淆矩阵
def print_confusion_matrix(cm):
    # 专门用来输出混淆矩阵的
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    print('\n'.join(lines))

# 打印出评分。
def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(round(score*100/best_score,4)) + "%)")
    return round(score*100/best_score,4)

# 输出最后的结果
print("*" * 80)
print('the result is :\n')
report_score(y_val_temp_t,y_val_temp_p)
print("*" * 80)

# 最后测试集的结果
y_test_temp = model.predict([test_head_data, test_body_data])
y_test_temp_p = np.argmax(y_test_temp,axis = 1)
y_test_temp_p = [LABELS[i] for i in y_test_temp_p]

# 保存结果，提交
df_test['Stance'] = y_test_temp_p
df_test.to_csv("../result/sub.csv",index = None)

