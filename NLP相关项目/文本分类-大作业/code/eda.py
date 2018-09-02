import pandas as pd 
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


print("*" * 50)
df = pd.read_pickle("../temp/df_train.pkl")
contents = df['cut_words'].tolist()
temp = []
for i in contents:
    temp.extend(i.split(' '))
word_map = Counter(temp)
word_map = sorted(word_map.items(),key = lambda x : x[-1],reverse = True)
words = [i[0] for i in word_map]
nums = [i[1] for i in word_map]

from pyecharts import WordCloud

wc = WordCloud(width=1300, height=620)
wc.add("词云图", words[:500], nums[:500],word_size_range=[10, 60],shape = 'star')
wc.render("../temp/词云图.html")

from pyecharts import Bar

bar = Bar(width=1300, height=620)
bar.add("高频词",words[1:10],nums[1:10])
bar.render("../temp/高频词bar.html")
print("EDA is OK.")
print("*" * 50)




