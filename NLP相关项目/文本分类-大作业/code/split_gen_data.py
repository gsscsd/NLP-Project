import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import gensim
import logging

df = pd.read_csv("../data/train.csv")
print(df.y.value_counts())
print(df.shape)
# df_pos = df[df['y'] == 1]
# df_neg = df[df['y'] == -1]

# df_pos = df_pos.sample(df_neg.shape[0] * 10)

# df_other = pd.concat([df_neg,df_pos])
# df_other.to_csv("../data/train.csv",index = None,encoding = 'utf-8')