import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def loadData():
    ids = np.load("idsMatrix.npy")
    labels = np.load("labelsMatrix.npy")
    # 打乱数据
    shuffle_index = np.random.permutation(ids.shape[0])
    ids = ids[shuffle_index]
    labels = labels[shuffle_index]
    x_train,x_valid,y_train,y_valid=train_test_split(ids,labels,test_size=0.1,random_state=2019)
    return x_train,x_valid,y_train,y_valid

#得到训练样本
def get_batch(inputs,targets,batch_size,shuffle=True):
    batch=[]
    if shuffle:
        index_shuffle=np.random.permutation(inputs.shape[0])
        inputs=inputs[index_shuffle]
        targets=targets[index_shuffle]
    counts=int(len(inputs)/batch_size)
    for i in range(counts):
        batch_x=inputs[i*batch_size:batch_size*(i+1)]
        batch_y=targets[i*batch_size:batch_size*(i+1)]
        batch.append((batch_x,batch_y))
    # if len(inputs)/batch_size!=0:
    #     batch_x=inputs[counts*batch_size:]
    #     batch_y=targets[counts*batch_size:]
    #     batch.append((batch_x, batch_y))
    return batch


if __name__=="__main__":
    x_train, x_valid, y_train, y_valid=loadData()
    print("input 测试集：",x_train.shape)
    print("inout 验证集：",x_valid.shape)
    print("target 测试集：",y_train.shape)
    print("target 验证集：",y_valid.shape)
    #加载词向量
    wordVector=np.load("wordVectors.npy")
    print("词向量shape:",wordVector.shape)

    ##构建模型
    #1.定义超参数
    batch_size=64 #参与训练样本数目大小
    n_hidden=128 #隐藏神经元个数
    num_classes=2 #标签个数
    epoch=20 #迭代次数
    max_seq_len=300 #序列长度
    embedding_size=50 #词向量维度大小
    learning_rate=0.001 #学习率大小

    input_data=tf.placeholder(dtype=tf.int32,shape=[batch_size,max_seq_len],name="input_data")
    x=tf.Variable(dtype=tf.float32,initial_value=tf.zeros([batch_size,max_seq_len,embedding_size]),name="word_embedding")
    x=tf.nn.embedding_lookup(wordVector,input_data)
    y=tf.placeholder(tf.float32,shape=[batch_size,num_classes],name="targets")

    weight=tf.get_variable("weight",shape=[n_hidden,num_classes],dtype=tf.float32,initializer=tf.random_normal_initializer)
    bias=tf.get_variable("bias",shape=[num_classes],dtype=tf.float32,initializer=tf.random_normal_initializer)

    lstmcell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=1.0,state_is_tuple=True)
    outputs,state=tf.nn.dynamic_rnn(lstmcell,x,dtype=tf.float32)
    outputs=tf.transpose(outputs,[1,0,2])
    pred=tf.matmul(outputs[-1],weight)+bias

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    index=np.random.randint(0,2250-64)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            total_loss=0
            for x_batch,y_batch in get_batch(x_train,y_train,batch_size=64):
                opt,l=sess.run([optimizer,loss],feed_dict={input_data:x_batch,y:y_batch})
                total_loss+=l
            valid_l,acc=sess.run([loss,accuracy],feed_dict={input_data:x_valid[index:index+64],y:y_valid[index:index+64]})
            print("epoch:{},训练集:loss={:.2f}，验证集:loss={:.2f},acc={}".format(i,total_loss,valid_l,acc))