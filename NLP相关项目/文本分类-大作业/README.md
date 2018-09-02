# 项目说明

## 一.报告说明

**另外一个pdf文件在report文件夹下面，包含了文本分类的基本方法与理论，以及实验的结果。**

## 二.项目结构

![](http://owzdb6ojd.bkt.clouddn.com/18-9-2/29265900.jpg)

如上图所示。code是代码文件夹，data是数据文件夹，temp是中间文件缓存文件夹，report是报告。

## 三.代码运行步骤

1. 首先在report文件夹下执行 :

   ```python
   pip install -r requiement.txt
   ```

   ​

2. 在code文件夹下执行

   ```shell
   python pre.py
   ```

3. 接着可以运行:

   ```
   python eda.py
   ```

   会在temp文件夹生成两个html，分别是词云和高频词的分布图。

4. 执行：

   ```
   python tfidf+rf+bnb.py bnb #分类器使用朴素贝叶斯
   python tfidf+rf+bnb.py rf #分类器使用随机森林
   ```

   用传统的方法来做文本分类

5. 执行:

   ```
   python biGru.py
   ```

   用双向GRU结构做文本分类。

6. 执行:

   ```
   python lstm+attention.py
   ```

   用TextRNN+Attention的方法做文本分类。

