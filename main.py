import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Model import Dynamic_LSTM, BiLSTMandAttention, TextCNN, RCNN, CNN_GRU, Transformer, Capsule
import time
import os

# tensorflow使用AVX2,提高运算速度
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def loadData():
    ids = np.load("idsMatrix.npy")
    labels = np.load("labelsMatrix.npy")
    # 打乱数据
    shuffle_index = np.random.permutation(ids.shape[0])
    ids = ids[shuffle_index]
    labels = labels[shuffle_index]
    x_train, x_valid, y_train, y_valid = train_test_split(ids, labels, test_size=0.1, random_state=2019)
    return x_train, x_valid, y_train, y_valid


# 得到训练样本
def get_batch(inputs, targets, batch_size, shuffle=True):
    batch = []
    if shuffle:
        index_shuffle = np.random.permutation(inputs.shape[0])
        inputs = inputs[index_shuffle]
        targets = targets[index_shuffle]
    counts = int(len(inputs) / batch_size)
    for i in range(counts):
        batch_x = inputs[i * batch_size:batch_size * (i + 1)]
        batch_y = targets[i * batch_size:batch_size * (i + 1)]
        batch.append((batch_x, batch_y))
    # if len(inputs) / batch_size != 0:
    #     batch_x = inputs[counts * batch_size:]
    #     batch_y = targets[counts * batch_size:]
    #     batch.append((batch_x, batch_y))
    return batch


def save_result(content):
    """将测试集结果保存到文本"""
    with open("./result.txt", 'a', encoding="utf8") as outp:
        outp.write(content + '\n')


def test(inputs,targets,path):
    model = CNN_GRU(n_hidden=128,
                    num_classes=2,
                    epoch=1,
                    max_seq_len=300,
                    embedding_size=50,
                    learning_rate=0.0005,
                    vocab_size=wordVector.shape[0],
                    num_layers=1,
                    wordVector=wordVector,
                    preEmbedding=True,
                    filter_sizes={1, 2, 3, 4},
                    num_filters=256)
    model.build_net()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        save_path=tf.train.latest_checkpoint(path)
        saver.restore(sess, save_path)
        iter=0
        p,r,f=0,0,0
        for x_test,y_test in get_batch(inputs,targets,batch_size=64):
            feed_dict={model.input_data: x_test, model.y: y_test, model.keep_prob: 1.0}
            prediction=sess.run(model.pred,feed_dict=feed_dict)
            _p,_r,_f=model.get_metrics(prediction,y_test)
            p+=_p
            r+=_r
            f+=_f
            iter+=1
        print("测试集结果：p={:.3f}，r={:.3f}，f={:.3f}\n".format(p/iter,r/iter,f/iter))


def train(path):
    x_train, x_valid, y_train, y_valid = loadData()
    print("input 测试集：", x_train.shape)
    print("inout 验证集：", x_valid.shape)
    print("target 测试集：", y_train.shape)
    print("target 验证集：", y_valid.shape)
    # 加载词向量
    wordVector = np.load("wordVectors.npy")
    print("词向量shape:", wordVector.shape)

    model = Capsule(max_seq_len=300,
                    embedding_size=50,
                    num_classes=2,
                    vocab_size=wordVector.shape[0],
                    wordVector=wordVector,
                    preEmbedding=True,
                    filter_size=3,
                    num_filters=32,
                    learning_rate=5e-4,
                    epoch=20)
    saver=tf.train.Saver(tf.global_variables(), max_to_keep=5)
    globalStep = tf.train.get_or_create_global_step()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(model.epoch):
            start = time.time()
            currentStep=tf.train.global_step(sess,globalStep)
            j = 0
            for x_batch, y_batch in get_batch(x_train, y_train, batch_size=64):
                feed_dict = {model.input_data: x_batch, model.y: y_batch, model.keep_prob: 0.5}
                opt, train_loss, prediction = sess.run([model.train_op, model.loss, model.pred], feed_dict=feed_dict)
                if j % 10 == 0:
                    p, r, f = model.get_metrics(prediction, y_batch)
                    print("epoch:{}, iter:{}, loss={:.3f}, p={:.3f}, r={:.3f}, f={:.3f}".format(i, j, train_loss, p, r,
                                                                                                f))
                j += 1
            iter, p, r, f = 0, 0, 0, 0
            for dev_x, dev_y in get_batch(x_valid, y_valid, batch_size=64):
                feed_dict = {model.input_data: dev_x, model.y: dev_y, model.keep_prob: 1.0}
                val_pred = sess.run(model.pred, feed_dict=feed_dict)
                _p, _r, _f = model.get_metrics(val_pred, dev_y)
                p += _p
                r += _r
                f += _f
                iter += 1
            end = time.time()
            print("验证集结果>>")
            content = "epoch:{}, p={:.3f}, r={:.3f}, f={:.3f},spend time={:.2f}".format(i, p / iter, r / iter, f / iter,
                                                                                        end - start)
            save_result(content)
            print(content+"\n")

            #模型保存
            saver.save(sess, save_path=path, global_step=currentStep)
            # checkpoint=tf.train.latest_checkpoint()
            print("Saved model checkpoint to {}\n".format(path))



if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    path = "./model/Capsule/my_model"  #模型保存地址
    train(path)
