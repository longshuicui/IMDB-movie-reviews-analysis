数据存放地址acllmdb文件夹 
程序仅包括训练过程，所用数据集为acllmdb/train
数据预处理过程 data_preprocess.py
数据转ids feature.py  结果有idsMatrix.npy  labelsMatrix.npy 
wordsList.npy 是词典，wordVectors.npy是预训练的词向量

模型选取：（网络层数均为单层！！！）
静态LSTM
动态LSTM          p=0.823, r=0.825, f=0.824
双向静态LSTM
双向动态LSTM+attention  p=0.847, r=0.860, f=0.853
双向动态LSTM
TextCNN
CNN+RNN结合 （CNN提取特征，RNN处理时序信息）

训练好的模型存放于model中

