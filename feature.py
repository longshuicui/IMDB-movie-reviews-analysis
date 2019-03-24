import numpy as np
import pandas as pd
import re
import os

##导入数据
wordList=np.load('wordsList.npy')
wordList=[word.decode('utf8') for word in wordList.tolist()]

##生成索引矩阵
strip_special_chars=re.compile(r"[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string=string.lower().replace("<br />",' ')
    return re.sub(strip_special_chars,'',string.lower())

max_seq_num=300 #最大序列数
num_files=25000 #文件数
ids=np.zeros((num_files,max_seq_num),dtype='int32')
file_count=0
pos_files=os.listdir("aclImdb/train/pos/")
neg_files=os.listdir("aclImdb/train/neg/")
for name in pos_files:
    path="aclImdb/train/pos/"+name
    indexCounter=0
    with open(path,'r',encoding='utf8') as file:
        content=file.read()
        cleanedLine=cleanSentences(content)
        split=cleanedLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter]=wordList.index(word)
            except ValueError:
                ids[file_count][indexCounter]=399999 #未知词
            indexCounter+=1
            if indexCounter>=max_seq_num:
                break
        file_count+=1

for name in neg_files:
    path = "aclImdb/train/neg/" + name
    indexCounter=0
    with open(path,'r',encoding='utf8') as file:
        content=file.read()
        cleanedLine=cleanSentences(content)
        split=cleanedLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter]=wordList.index(word)
            except ValueError:
                ids[file_count][indexCounter] = 399999  # 未知词
            indexCounter+=1
            if indexCounter>=max_seq_num:
                break
        file_count+=1

np.save('idsMatrix',ids)
print(ids)
