#确定并设置最好的序列长度

import os
import matplotlib.pyplot as plt

pos_path=r"aclImdb\train\pos/"
pos_list=os.listdir(pos_path)
num_words=[]
for name in pos_list:
    file_path=pos_path+name
    with open(file_path,encoding='utf8') as file:
        content=file.read()
        count=len(content.split())
        num_words.append(count)
print('正例统计完毕')
neg_path=r"aclImdb\train\neg/"
neg_list=os.listdir(neg_path)
for name in neg_list:
    file_path=neg_path+name
    with open(file_path,encoding='utf8') as file:
        content=file.read()
        count=len(content.split())
        num_words.append(count)
print('负例统计完毕')

plt.figure()
plt.hist(num_words,50,color='g')
plt.show()

print('文档总数',len(num_words))
print('单词总数',sum(num_words))
print('平均单词数量',sum(num_words)/len(num_words))

"""
正例统计完毕
负例统计完毕
文档总数 25000
单词总数 5844680
平均单词数量 233.7872
"""
