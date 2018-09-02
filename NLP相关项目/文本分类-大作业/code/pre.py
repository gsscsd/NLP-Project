import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import jieba
from gensim.models import word2vec
import gensim
import logging
import warnings

warnings.filterwarnings('ignore')


# 主函数
def main():
    # 加载训练集和测试集
    print("*" * 80)
    print("start .....")
    dir_0 = "../data/train.csv"
    df_0 = pd.read_csv(dir_0)

    pre(df_0,'df_train')
    save_text(df_0,'df_cut_words')

    save_word2vec("../temp/" + "df_cut_words.csv",'../temp/word2vec.vec',300)
    print("*" * 80)
    pass


# 预处理函数，分词并保存
def pre(df,name):
    print(name,"is cut words")
    stopwords = load_stop_words()
    df['cut_words'] = df['Discuss'].map(lambda x : " ".join([i for i in list(jieba.cut(x)) if i not in stopwords]))
    df.to_pickle("../temp/" + name + '.pkl')
    print(name,'cut words is Ok.')

def save_text(train,name):
    print("save text.")
    df = train
    df['cut_words'].to_csv("../temp/" + name + ".csv",index = None,encoding = 'utf-8')
    print("It is OK.")

# word2vec编译
def save_word2vec(cut_txt,save_model_name,n_dim = 300):
    if not os.path.exists(save_model_name):     # 判断文件是否存在
        model_train(cut_txt, save_model_name,n_dim)
    else:
        print('此训练模型已经存在，不用再次训练')
    pass

##word2vec模型训练
def model_train(train_file_name, save_model_file,n_dim = 300):  # model_file_name为训练语料的路径,save_model为保存模型名
    """
    train_file_name : 训练文件路径
    save_model_file : 保存模型路径
    n_dim : 特征维度,默认300维度
    """
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=n_dim)  # 训练skip-gram模型; 默认window=5
    model.save(save_model_file)
#     model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用###转换词向量


# 加载停用词
def load_stop_words(path = "../data/stopwords.txt"):
    """
    加载停用词并返回
    """
    stopwords = []
    with open(path,'r',encoding='utf-8') as f :
        for i in f.readlines():
            stopwords.append(i.strip())
    return stopwords




if __name__ == '__main__':
    main()