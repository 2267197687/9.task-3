import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec

 
f = open("sanguo.txt", 'r',encoding='utf-8') 
lines = []
for line in f: #分别对每段分词
    temp = jieba.lcut(line)  #分词
    words = []
    for i in temp:
        #过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])
 
 
# 调用Word2Vec训练
# 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines,vector_size = 20, window = 2 , min_count = 3, epochs=7, negative=10,sg=1)
print("周瑜的词向量：\n",model.wv.get_vector('周瑜'))
a = model.wv.most_similar('周瑜', topn = 20)# 与周瑜最相关的前20个词语
print(f"\n和周瑜相关性最高的前20个词语:{a}")
 
words = model.wv.most_similar(positive=['孔明', '司马懿'], negative=['周瑜'])#类比孔明-司马懿 = 周瑜 -？（挺好玩的
print(words)
 
