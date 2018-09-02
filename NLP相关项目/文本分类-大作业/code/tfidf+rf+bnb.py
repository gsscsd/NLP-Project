import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.sparse import csr_matrix, hstack
import jieba
import sys
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import BernoulliNB
import warnings

warnings.filterwarnings('ignore')



start_time = time.time()


# 主函数，加载分词后的数据
def main():
    name = sys.argv[1]
    df = pd.read_pickle("../temp/df_train.pkl")
    train = tfidf_vec(df)
    y = df['y'].values
    N = 6
    rf = RandomForestClassifier()
    bnb = BernoulliNB()
    if name == 'rf':
        accs,pres,recs = cv_model(rf,'RandomForestClassifier',N,train,y)
        print("accs ",accs)
        print("pres ",pres)
        print("recs ",recs)
        visual_r_a_p(accs,pres,recs,N)
    elif name == 'bnb':
        accs,pres,recs = cv_model(bnb,'BernoulliNB',N,train,y)
        print("accs ",accs)
        print("pres ",pres)
        print("recs ",recs)
        visual_r_a_p(accs,pres,recs,N)
    else:
        print("Input Error,Please Input rf or bnb.")

###根据名字，选择cv
def cv_model(model,name,N,X,y):
    kf = KFold(n_splits=N,shuffle=False,random_state=42)
    kf = kf.split(X)
    pres = []
    recs = []
    accs = []
    ###kf训练
    for i ,(train_fold,test_fold) in enumerate(kf):
        X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
        model.fit(X_train, label_train)
        val_ = model.predict(X=X_validate)
        print(i,name,'acc is ',accuracy_score(label_validate,val_))
        print(i,name,'pre is ',precision_score(label_validate,val_))
        print(i,name,'rec is ',recall_score(label_validate,val_))
        pres.append(precision_score(label_validate,val_))
        recs.append(recall_score(label_validate,val_))
        accs.append(accuracy_score(label_validate,val_))
    return accs,pres,recs

# tfidf词向量
def tfidf_vec(df):
    tf = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
    discuss_tf = tf.fit_transform(df['cut_words'])
    return discuss_tf


def visual_r_a_p(accs,pres,recs,N):
    plt.figure()
    plt.plot(np.arange(N),accs,color="green",label = 'accuracy')
    plt.plot(np.arange(N),pres,color="red",label = 'precision')
    plt.plot(np.arange(N),recs,color="blue",label = 'recall')
    plt.title("acc-pre-recall")
    plt.xlabel("N")
    plt.ylabel("prob")
    plt.legend(['accuracy','precision','recall'])
    plt.show()
    pass

if __name__ == '__main__':
    main()
    
    