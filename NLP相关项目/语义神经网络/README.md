### 语义神经网络

---

一.序言：

> 在NLP领域，语义相似度的计算一直是个难题：搜索场景下query和Doc的语义相似度、feeds场景下Doc和Doc的语义相似度、机器翻译场景下A句子和B句子的语义相似度等等。本项目实现LSTM-DSSM模型在Fake News Challenge做测试。

二.实验内容:

> 本次实验采用的数据集是FAKE NEWS CHALLENGE STAGE 1：STANCE DETECTION，本次实验选择了从新闻文章相对于标题来估计正文文本的立场的任务。具体地说，正文文本可以同意、不同意、讨论或与标题无关。

三.模型结构：

![](http://owzdb6ojd.bkt.clouddn.com/18-9-2/60604421.jpg)

三.实验结果：

训练过程中的准确与损失：

![](http://owzdb6ojd.bkt.clouddn.com/18-9-2/11375357.jpg)

以及最后的结果：

![](http://owzdb6ojd.bkt.clouddn.com/18-9-2/52303131.jpg)

与使用GBDT算法的baseline相比，有了比较大的提升。

附：官网提供的[baseline](https://github.com/FakeNewsChallenge/fnc-1-baseline)