import tensorflow as tf
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class TextClassifier(object):
    """模型超参数"""

    def __init__(self,
                 n_hidden=128,
                 num_classes=2,
                 epoch=20,
                 max_seq_len=300,
                 embedding_size=50,
                 learning_rate=0.0005,
                 vocab_size=None,
                 num_layers=1,
                 wordVector=None,
                 preEmbedding=True,
                 isAttention=False,
                 filter_sizes=None,
                 num_filters=256,
                 num_blocks=2,
                 head=8):
        if filter_sizes is None:
            filter_sizes = [1, 2, 3, 4]
        self.n_hidden = n_hidden  # LSTM单元隐藏层神经元个数
        self.num_classes = num_classes  # 标签个数
        self.epoch = epoch  # 训练次数
        self.max_seq_len = max_seq_len  # 序列最大长度
        self.embedding_size = embedding_size  # 词向量维度
        self.learning_rate = learning_rate  # 学习率
        self.vocab_size = vocab_size  # 词典大小，embedding嵌入用
        self.num_layers = num_layers  # LSTM网络层数
        self.isAttention = isAttention  # biLSTM中是否用attention机制
        self.filter_sizes = filter_sizes  # 用于textCNN 卷积核大小
        self.num_filters = num_filters  # 用于textCNN 卷积核数量
        self.output_size = 128  # 用于RCNN模型中biLSTM 的输出维度大小
        self.num_blocks = num_blocks  # transformer encoder block的个数
        self.head = head  # multi-attention 的头数

        # 定义模型输入输出
        self.input_data = tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name="input_data")
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="target")
        with tf.name_scope("embedding"):
            embedding = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.embedding_size], dtype=tf.float32),
                                    name="word_embedding")
            if preEmbedding:
                embedding = embedding.assign(tf.cast(wordVector, tf.float32))
                tf.logging.info("已使用预训练的词向量")
        self.x = tf.nn.embedding_lookup(embedding, self.input_data)
        self.keep_prob = tf.placeholder(name="keep_prob", dtype=tf.float32)
        # print(self.x.shape)  # (?, 300, 50)
        with tf.name_scope("build_net"):
            self.build_net()

    def build_net(self):
        pass

    def get_metrics(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, 1)
        y_true = np.argmax(y_true, 1)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        return p, r, f


class Dynamic_LSTM(TextClassifier):

    def build_net(self):
        # 定义softmax层参数
        weight = tf.get_variable("weight", shape=[self.n_hidden, self.num_classes], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer)
        bias = tf.get_variable("bias", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer)

        # 定义lstm层
        lstmcell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstmcell = tf.nn.rnn_cell.DropoutWrapper(lstmcell, output_keep_prob=self.keep_prob)
        lstmcell = tf.nn.rnn_cell.MultiRNNCell([lstmcell] * self.num_layers)
        outputs, states = tf.nn.dynamic_rnn(lstmcell, self.x, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        output = outputs[-1]
        output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        self.pred = tf.matmul(output, weight) + bias
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)


class BiLSTMandAttention(TextClassifier):
    """双向LSTM+Attention"""

    def build_net(self):
        # self.x=tf.unstack(self.x,self.max_seq_len,1)
        tf.logging.info("the shape of input tensor:%s" % str(self.x.shape))
        # 定义softmax层参数

        with tf.name_scope("bilstm"):
            def cell_fw():
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            def cell_bw():
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            stack_fw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_fw() for _ in range(self.num_layers)],
                                                        state_is_tuple=True)
            stack_bw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_bw() for _ in range(self.num_layers)],
                                                        state_is_tuple=True)
            # outputs是正向和反向输出的元组,states是正向和反向最后时刻状态的元组
            outputs, states = tf.nn.bidirectional_dynamic_rnn(stack_fw_cell, stack_bw_cell, self.x, dtype=tf.float32)
        # 是否添加attention机制
        if self.isAttention:
            output = outputs[0] + outputs[1]
            with tf.name_scope("attention"):
                output = self.attention(output)  # 将各个时刻的状态值加权求和，结果做softmax分类
            weight = tf.get_variable("weight", shape=[self.n_hidden, self.num_classes], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            bias = tf.get_variable("bias", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer)
        else:
            output = tf.concat([outputs[0], outputs[1]], axis=2)
            tf.logging.info("output:%s" % str(output.shape))
            output = tf.transpose(output, [1, 0, 2])
            output = output[-1]  # 将最后一个时间步做softmax分类
            tf.logging.info("output:%s" % str(output.shape))
            weight = tf.get_variable("weight", shape=[self.n_hidden * 2, self.num_classes], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer)
            bias = tf.get_variable("bias", shape=[self.num_classes], dtype=tf.float32, initializer=tf.zeros_initializer)

        with tf.name_scope("outputs"):
            self.pred = tf.matmul(output, weight) + bias

        with tf.name_scope("loss_train_op"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def attention(self, H):
        # 初始化一个可训练权重向量，
        w = tf.Variable(tf.random_normal([self.n_hidden], stddev=0.1), name="attention_w")
        # 对输出做非线性转换
        M = tf.tanh(H)
        # 对w和M做矩阵运算，w=[batch_size,time_step,n_hidden],维度转换[batch_size*time_step,n_hidden]
        # 每一个时间步的输出向量转换为一个数字
        newM = tf.matmul(tf.reshape(M, [-1, self.n_hidden]), tf.reshape(w, [-1, 1]))
        # 对newM维度转换成[batch_size,time_step]
        restoreM = tf.reshape(newM, [-1, self.max_seq_len])
        # softmax归一化处理[batch_size,time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        H = tf.transpose(H, [0, 2, 1])
        r = tf.matmul(H, tf.reshape(self.alpha, [-1, self.max_seq_len, 1]))
        # 将三维压缩成二维sequeeze
        sequeezeR = tf.squeeze(r)
        output = tf.tanh(sequeezeR)
        # 做dropout处理
        output = tf.nn.dropout(output, keep_prob=self.keep_prob)
        return output


class TextCNN(TextClassifier):
    """TextCNN模型"""

    def build_net(self):
        # 扩展输入的维度
        self.x = tf.expand_dims(self.x, axis=-1, name="expand_dim")
        pooling_res = []
        for filter_size in self.filter_sizes:
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
            weight = tf.Variable(tf.random_normal(shape=filter_shape, stddev=1.0), dtype=tf.float32)
            bias = tf.Variable(tf.random_normal(shape=[self.num_filters], stddev=1.0), dtype=tf.float32)
            conv = tf.nn.conv2d(self.x, weight, [1, 1, 1, 1], "VALID", name="conv")
            relu = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            max_pool = tf.nn.max_pool(relu, ksize=[1, self.max_seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                      padding="VALID", name="maxpool")
            pooling_res.append(max_pool)
        # 输入全连接层的神经元的个数
        num_fc = self.num_filters * len(self.filter_sizes)
        # print(pooling_res[0].shape)  # max_pool的shape (?, 1, 1, 256)
        fc_input = tf.concat(pooling_res, axis=3)  # 按照最后一个维度合并
        # print(fc_input.shape)  # 合并之后的shape (?, 1, 1, 1024)
        fc_input = tf.reshape(fc_input, shape=[-1, num_fc])
        # print(fc_input.shape)  # reshape之后的shape (?, 1024)
        fc_input = tf.nn.dropout(fc_input, keep_prob=self.keep_prob)
        with tf.name_scope("output"):
            w = tf.Variable(tf.truncated_normal(shape=[num_fc, self.num_classes], stddev=1.0), dtype=tf.float32,
                            name="w")
            b = tf.Variable(tf.zeros(shape=[self.num_classes]), name='b')
            self.pred = tf.matmul(fc_input, w) + b
        with tf.name_scope("loss_train"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)


class RCNN(TextClassifier):
    """
    RNN+CNN 模型
    1.将双向LSTM的正向输出和反向输出和词向量拼成一个上下文向量，
    2.经tanh函数激活，送入最大池化层，
    3.然后将池化结果传入全连接做分类。
    """

    def build_net(self):
        with tf.name_scope("bilstm"):
            def cell_fw():
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            def cell_bw():
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
                return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

            stack_fw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_fw() for _ in range(self.num_layers)],
                                                        state_is_tuple=True)
            stack_bw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_bw() for _ in range(self.num_layers)],
                                                        state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(stack_fw_cell, stack_bw_cell, self.x,
                                                                             dtype=tf.float32)

        with tf.name_scope("context"):
            tf.logging.info("the shape of foreward vector:%s" % str(output_fw.shape))  # (?, 300, 128)
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            # 获得上文信息 (?, 300, 128)[-1,max_seq_len,n_hidden]
            context_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="contextleft")
            # 获得下文信息 (?, 300, 128)[-1,max_seq_len,n_hidden]
            context_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="contextright")
            # 将前向/后向和当前词向量拼接在一起，作为最终的表征向量 (?, 300, 306)[-1,max_seq_len,2*n_hidden+embedding_size]
            context = tf.concat([context_left, self.x, context_right], axis=2, name="context")
            tf.logging.info("the shape of context_left:%s" % str(context_left.shape))
            tf.logging.info("the shape of context_right:%s" % str(context_right.shape))
            tf.logging.info("the shape of context%s" % str(context.shape))
        with tf.name_scope("rnn_output"):
            size = self.n_hidden * 2 + self.embedding_size
            weight = tf.Variable(tf.random_normal(shape=[size, self.output_size], stddev=1.0), dtype=tf.float32,
                                 name="rnn_output_weight")
            bias = tf.Variable(tf.random_normal(shape=[self.output_size]), dtype=tf.float32, name="rnn_output_bias")
            rnn_output = tf.tanh(tf.einsum("aij,jk->aik", context, weight) + bias)  # tf.einsum()方法用于指定维度消除
            tf.logging.info("the shape of rnn_output:%s" % str(rnn_output.shape))  # (?, 300, 128)
        with tf.name_scope("max_pool"):
            max_pool = tf.reduce_max(rnn_output, axis=1, name="max_pooling")
            tf.logging.info("the shape of max_pool output:%s" % str(max_pool.shape))  # (?, 128)将时间步的维度消除，取每个时间步最大的值
        with tf.name_scope("softmax"):
            w = tf.Variable(tf.truncated_normal([self.output_size, self.num_classes], stddev=1.0), dtype=tf.float32,
                            name="w")
            b = tf.Variable(tf.truncated_normal([self.num_classes]), dtype=tf.float32, name="b")

            pred = tf.matmul(max_pool, w) + b
            self.pred = tf.nn.softmax(pred)
        with tf.name_scope("loss_train"):
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            # globalStep = tf.Variable(0, name="globalStep", trainable=False)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)) + 0.5 * l2_loss  # 增加L2正则
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # gradAndtvar = optimizer.compute_gradients(self.loss)  # 获得梯度和变量
            # self.train_op = optimizer.apply_gradients(gradAndtvar, global_step=globalStep)
            self.train_op = optimizer.minimize(self.loss)


class CNN_GRU(TextClassifier):
    """CNN+GRU网络模型 将两者提取的信息concat，进行分类"""
    def build_net(self):
        with tf.name_scope("cnn"):
            x = tf.expand_dims(self.x, axis=-1, name="expand_dim")
            pooling_res = []
            for filter_size in self.filter_sizes:
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                weight = tf.Variable(tf.truncated_normal(shape=filter_shape), dtype=tf.float32, name="weight")
                bias = tf.Variable(tf.zeros(shape=[self.num_filters]), dtype=tf.float32, name="bias")
                conv = tf.nn.conv2d(x, filter=weight, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
                max_pool = tf.nn.max_pool(relu, ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                                          strides=[1, 1, 1, 1], padding="VALID", name="pooling")
                pooling_res.append(max_pool)
            num_fc = self.num_filters * len(self.filter_sizes)
            fc_input = tf.concat(pooling_res, axis=3)
            cnn_output = tf.reshape(fc_input, shape=[-1, num_fc])
        with tf.name_scope("GRU"):
            cell = tf.nn.rnn_cell.GRUCell(self.n_hidden)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            rnn_output = outputs[-1]
        with tf.name_scope("softmax"):
            fc = tf.concat([cnn_output, rnn_output], 1)
            fc = tf.nn.dropout(fc, keep_prob=self.keep_prob)
            w = tf.Variable(tf.truncated_normal(shape=[self.n_hidden + num_fc, self.num_classes]), dtype=tf.float32,
                            name="w")
            b = tf.Variable(tf.truncated_normal(shape=[self.num_classes]), dtype=tf.float32, name="b")
            self.pred = tf.matmul(fc, w) + b
        with tf.name_scope("loss_train"):
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y)) + 0.5 * l2_loss  # 增加L2正则
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss)


class Transformer(TextClassifier):
    """
    transfromer encoder 用于文本分类
    1.position embedding，用one-hot，数据量小
    2.block层数设置为2
    3.sublayer加dropout正则化
    4.前馈层用卷积实现（全连接实现）待定
    """

    def _onehot_position_embedding(self, batch_size=64, max_len=300):
        """采取one-hot编码生成位置向量"""
        position_embedding = []
        for batch in range(batch_size):
            x = []
            for step in range(max_len):
                vector = np.zeros(max_len)
                vector[step] = 1
                x.append(vector)
            position_embedding.append(x)
        return np.array(position_embedding, dtype=np.float32)

    def _position_embedding(self):
        """采用论文中sin/cos生成位置向量"""
        batch_size = tf.shape(self.x)[0]
        # 生成位置索引，并扩张到所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(self.max_seq_len), 0), [batch_size, 1])
        # 根据正弦和余弦函数获得每个位置上的第一部分
        positionEmbedding = np.array(
            [[pos / np.power(10000, (i - i % 2) / self.embedding_size) for i in range(self.embedding_size)] for pos in
             range(self.max_seq_len)])
        # 根据奇偶性分别用sin和cos函数包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])
        # 转换成tensor格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)
        # 获得三维的矩阵
        positionEmbedding = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)
        return positionEmbedding

    def _multihead_attention(self,raw_inputs,queries,keys,size=None,causality=False):
        """
        参数
        :param raw_inputs:  原始输入，用于计算mask
        :param queries: 用于计算相关度，与key相同则是同一句子计算相关性
        :param keys: 用于计算相关度，word_embedding+position_embedding
        :param size: dense全连接层输出维度大小
        :param causality:
        :return:
        """
        if size is None:
            size=queries.shape[-1]
        #对数据分割映射(论文中是先分割在映射)
        Q=tf.layers.dense(queries,size,activation=tf.nn.relu)
        K=tf.layers.dense(keys,size,activation=tf.nn.relu)
        V=tf.layers.dense(keys,size,activation=tf.nn.relu)
        Q_=tf.concat(tf.split(Q,self.head,axis=-1),axis=0)
        K_=tf.concat(tf.split(K,self.head,axis=-1),axis=0)
        V_=tf.concat(tf.split(V,self.head,axis=-1),axis=0)
        #输出维度
        tf.logging.info("the shape of Q:%s"%str(Q.shape)) #(64, 300, 350)
        tf.logging.info("the shape of Q_:%s"%str(Q_.shape)) #(320, 300, 70)

        #计算k和q的点积，维度[batch_size*numheads,q_len,k_len]
        similary=tf.matmul(Q_,tf.transpose(K_,[0,2,1]))
        #进行缩放处理
        scaled_similary=similary/(K_.get_shape().as_list()[-1]**0.5)
        #对填充词的处理，当padding为0是时，计算的权重应该是0，需要将其mask为0，则需要q或者k一方为0
        #将每一个时序上的向量中的值相加取平均值
        key_mask=tf.sign(tf.abs(tf.reduce_sum(raw_inputs,axis=-1))) #[batch_size,time_step]
        #利用tf.tile进行张量扩张，维度[batch_size*numhead,key_len]
        key_mask=tf.tile(key_mask,[self.head,1])
        #增加一个维度，并进行扩张，维度[batch_size*numhead,q_len,k_len]
        key_mask=tf.tile(tf.expand_dims(key_mask,1),[1,queries.shape[1],1])
        tf.logging.info("the shape of key_mask:%s"%str(key_mask.shape)) #(?, 300, 300)

        #生成与scale_similary相同维度的tensor，然后得到负无穷大的值
        paddings=tf.ones_like(scaled_similary)*(-2**(32+1))
        #如果keymask中的值为0则用padding替换，
        masked_similary=tf.where(tf.equal(key_mask,0),paddings,scaled_similary)

        #计算权重系数
        weight=tf.nn.softmax(masked_similary)
        #加权求和，维度[batch_size*numheads,time_step,embedding_size/numheads]
        outputs=tf.matmul(weight,V_)
        #将outputs重组
        outputs=tf.concat(tf.split(outputs,self.head,0),axis=2)
        outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)

        #对每一个sublayer建立残差连接
        outputs+=queries
        #标准层
        outputs = self._layer_normalization(outputs)
        return outputs


    def _layer_normalization(self,inputs):
        """Batch Normalization层"""
        inputshape=inputs.shape
        paramshape=inputshape[-1]
        #LayerNorm是在最后的维度上计算输入的数据的均值和方差，维度[batch_size,time_step,1]
        mean,var=tf.nn.moments(inputs,[-1],keep_dims=True)

        beta=tf.Variable(tf.zeros(paramshape))
        gamma=tf.Variable(tf.ones(paramshape))
        normalized=(inputs-mean)/((var+1.0)**0.5)
        outputs=gamma*normalized+beta   #重构 
        return outputs


    def _feed_forward(self,inputs):
        filters=[128,self.embedding_size+self.max_seq_len] #[内层，外层]
        #前馈层采用卷积神经网络 1维卷积 实际的维度还是二维的[batch_size,time_step,embedding_size]
        param={"inputs":inputs,"filters":filters[0],"kernel_size":1,"activation":tf.nn.relu,"use_bias":True}
        outputs=tf.layers.conv1d(**param)
        param={"inputs":outputs,"filters": filters[1], "kernel_size": 1,"activation": None, "use_bias": True}
        outputs=tf.layers.conv1d(**param)

        #残差连接和归一化处理
        outputs+=inputs
        outputs=self._layer_normalization(outputs)
        return outputs


    def build_net(self):
        with tf.name_scope("input"):
            position_embedding = self._onehot_position_embedding(batch_size=64, max_len=self.max_seq_len)
            embedded_words = tf.concat([self.x, position_embedding], axis=-1)
            tf.logging.info("the shape of position embedding:%s" % str(position_embedding.shape))
            tf.logging.info("the shape of embedded words:%s" % str(embedded_words.shape))

        with tf.name_scope("encoder"):
            for i in range(self.num_blocks):
                with tf.name_scope("encoder-%d"%(i+1)):
                    multihead_attention=self._multihead_attention(raw_inputs=self.x,queries=embedded_words,keys=embedded_words)
                    embedded_words=self._feed_forward(inputs=multihead_attention)
            tf.logging.info("the last encoder block's shape:%s"%str(embedded_words.shape)) #(64, 300, 350)

            outputs=tf.reshape(embedded_words,[-1,self.max_seq_len*(self.embedding_size+self.max_seq_len)])
            output_size=outputs.shape[-1]

        with tf.name_scope("dropout"):
            outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            w=tf.get_variable("output_w",shape=[output_size,self.num_classes],initializer=tf.random_normal_initializer)
            b=tf.get_variable("output_b",shape=[self.num_classes],initializer=tf.zeros_initializer)
            self.pred=tf.matmul(outputs,w)+b

        with tf.name_scope("loss_train"):
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,labels=self.y))
            optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op=optimizer.minimize(self.loss)


class Capsule(object):
    """
    Capsule应用于文本分类，
    论文《Investigating Capsule Networks with Dynamic Routing for Text Classification》
    第一层：卷积层  卷积核K1=3，卷积核个数B=32
    第二层：基础capsule  卷积核个数C=32，输出g(W*Mi+b)
    """
    def __init__(self,
                 max_seq_len=300,
                 embedding_size=50,
                 num_classes=2,
                 vocab_size=None,
                 wordVector=None,
                 preEmbedding=True,
                 filter_size=3,
                 num_filters=32,
                 learning_rate=0.0005,
                 epoch=20):
        self.max_seq_len=max_seq_len  #最大序列长度
        self.embedding_size=embedding_size  #词向量维度大小
        self.num_classes=num_classes  #类别标签个数
        self.vocab_size=vocab_size  #embedding中词的个数
        self.filter_size=filter_size  #卷积核大小，论文中大小为3
        self.num_filters=num_filters  #卷积核数量
        self.lr=learning_rate
        self.epoch=epoch

        self.keep_prob=tf.placeholder(tf.float32,name="keep_prob")
        self.input_data = tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name="input_data")
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="target")
        with tf.name_scope("embedding"):
            embedding = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.embedding_size], dtype=tf.float32),
                                    name="word_embedding")
            if preEmbedding:
                embedding = embedding.assign(tf.cast(wordVector, tf.float32))
                tf.logging.info("已使用预训练的词向量")
        x = tf.nn.embedding_lookup(embedding, self.input_data)
        self.x=tf.expand_dims(x,-1)

        with tf.name_scope("build_net"):
            self.build_net()

    def build_net(self):
        with tf.name_scope("conv_layer"):
            filter_shape=[self.filter_size,self.embedding_size,1,self.num_filters]  #[3,50,1,32]
            w_conv=tf.Variable(tf.truncated_normal(shape=filter_shape),dtype=tf.float32,name="w_conv")
            b_conv=tf.Variable(tf.truncated_normal(shape=[self.num_filters]),dtype=tf.float32,name="b_conv")
            conv=tf.nn.conv2d(self.x,filter=w_conv,strides=[1,2,1,1],padding="VALID",name="conv")
            conv_output=tf.nn.relu(tf.nn.bias_add(conv,b_conv))
            #返回shape[batch_size,max_len-filter_size+1,1,num_filters]
            tf.logging.info("the shape of conv_output {}".format(conv_output.get_shape()))

        with tf.name_scope("primary_capsule_layer"):
            w_pc=tf.Variable(tf.truncated_normal(shape=[1,1,32,16]),dtype=tf.float32,name="w_pc")
            b_pc=tf.Variable(tf.truncated_normal(shape=[16]),dtype=tf.float32,name="b_pc")
            conv=tf.nn.conv2d(conv_output,filter=w_pc,strides=[1,1,1,1],padding="VALID",name="pc_conv")
            pose=tf.nn.relu(tf.nn.bias_add(conv,b_pc),name="pc_relu")
            pose_shape=pose.get_shape().as_list()
            pose=tf.reshape(pose,[-1, pose_shape[1], pose_shape[2], 16, 16])
            tf.logging.info("the shape of primary_pose {}".format(pose.get_shape())) #(?, 149, 1, 16, 16)
            pose=self.squash(pose)
            bias_2=tf.Variable(tf.truncated_normal(shape=[1,16]),dtype=tf.float32,name="bias_2")
            activation=tf.sqrt(np.sum(np.square(pose),axis=-1))+bias_2

        with tf.name_scope("capsule_conv_layer"):
            inputs_pose_shape=pose.get_shape().as_list()
            shape=[3,1,16,16]
            hk_offsets=[
                [int(h_offset+k_offset) for k_offset in range(0,shape[0])] for h_offset in range(0,inputs_pose_shape[1]+1-shape[0],1)
            ]
            wk_offsets = [
                [int(w_offset + k_offset) for k_offset in range(0, shape[1])] for w_offset in range(0, inputs_pose_shape[2] + 1 - shape[1], 1)
            ]
            print(hk_offsets)
            print(wk_offsets)
            inputs_poses_patches = tf.transpose(
                tf.gather(
                    tf.gather(
                        pose, hk_offsets, axis=1, name='gather_poses_height_kernel'
                    ), wk_offsets, axis=3, name='gather_poses_width_kernel'
                ), perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_poses_patches'
            )
            tf.logging.info('i_poses_patches shape: {}'.format(inputs_poses_patches.get_shape()))

            inputs_poses_shape = inputs_poses_patches.get_shape().as_list()
            inputs_poses_patches = tf.reshape(inputs_poses_patches, [
                -1, shape[0] * shape[1] * shape[2], inputs_poses_shape[-1]
            ])
            i_activations_patches = tf.transpose(
                tf.gather(
                    tf.gather(
                        activation, hk_offsets, axis=1, name='gather_activations_height_kernel'
                    ), wk_offsets, axis=3, name='gather_activations_width_kernel'
                ), perm=[0, 1, 3, 2, 4, 5, 6], name='inputs_activations_patches'
            )
            tf.logging.info('i_activations_patches shape: {}'.format(i_activations_patches.get_shape()))
            i_activations_patches = tf.reshape(i_activations_patches, [-1, shape[0] * shape[1] * shape[2]])
            u_hat_vecs = self.vec_transformationByConv(inputs_poses_patches,
                                                       inputs_poses_shape[-1],
                                                       shape[0] * shape[1] * shape[2],
                                                       inputs_poses_shape[-1],
                                                       shape[3])
            tf.logging.info('capsule conv votes shape: {}'.format(u_hat_vecs.get_shape()))
            beta_a=tf.Variable(tf.random_normal(shape=[1,shape[3]]),dtype=tf.float32,name="beta_a")
            poses, activations = self.routing(u_hat_vecs, beta_a, 3, shape[3], i_activations_patches)
            poses = tf.reshape(poses, [
                -1, inputs_poses_shape[1],
                inputs_poses_shape[2], shape[3],
                inputs_poses_shape[-1]]
                               )
            activations = tf.reshape(activations, [
                -1, inputs_poses_shape[1],
                inputs_poses_shape[2], shape[3]])
            tf.logging.info("capsule conv poses dimension:{}".format(poses.get_shape()))
            tf.logging.info("capsule conv activations dimension:{}".format(activations.get_shape()))

        with tf.name_scope("capsule_flatten"):
            input_pose_shape=poses.get_shape().as_list()
            poses = tf.reshape(poses, [
                -1, input_pose_shape[1] * input_pose_shape[2] * input_pose_shape[3], input_pose_shape[-1]])
            activations = tf.reshape(activations, [
                -1, input_pose_shape[1] * input_pose_shape[2] * input_pose_shape[3]])
            tf.logging.info("flatten poses dimension:{}".format(poses.get_shape()))
            tf.logging.info("flatten activations dimension:{}".format(activations.get_shape()))

        with tf.name_scope("capsule_fc_layer"):
            input_pose_shape=poses.get_shape().as_list()
            u_hat_vecs = self.vec_transformationByConv(
                poses,
                input_pose_shape[-1], input_pose_shape[1],
                input_pose_shape[-1], self.num_classes,
            )
            tf.logging.info('votes shape: {}'.format(u_hat_vecs.get_shape()))
            beta_b=tf.Variable(tf.random_normal(shape=[1,self.num_classes]),tf.float32,name="beta_b")
            poses, activations = self.routing(u_hat_vecs, beta_b, 3, self.num_classes,i_activations=None)
            tf.logging.info('capsule fc shape: {}'.format(poses.get_shape()))
            # print(activations)
            # exit()
        with tf.name_scope("loss_train"):
            self.pred=activations
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,labels=self.y))
            optimizer=tf.train.AdamOptimizer(learning_rate=self.lr)
            gradAndVar=optimizer.compute_gradients(self.loss)
            self.train_op=optimizer.apply_gradients(grads_and_vars=gradAndVar)


    def squash(self,x):
        """向量归一化，使长度小于1"""
        s_squared_norm = np.sum(np.square(x), axis=-1, keepdims=True)
        scale =tf.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
        return scale * x

    def vec_transformationByConv(self,poses, input_capsule_dim, input_capsule_num, output_capsule_dim, output_capsule_num):
        kernel=tf.Variable(tf.random_normal(shape=[1, input_capsule_dim, output_capsule_dim * output_capsule_num]),dtype=tf.float32,name="weight")
        tf.logging.info('poses: {}'.format(poses.get_shape()))
        tf.logging.info('kernel: {}'.format(kernel.get_shape()))
        u_hat_vecs = tf.nn.conv1d(poses, kernel,stride=1,padding="VALID")
        u_hat_vecs = tf.reshape(u_hat_vecs, (-1, input_capsule_num, output_capsule_num, output_capsule_dim))
        u_hat_vecs = tf.transpose(u_hat_vecs, (0, 2, 1, 3))
        return u_hat_vecs

    def routing(self, u_hat_vecs, beta_a, iterations, output_capsule_num, i_activations):
        from keras import backend as K
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])
        if i_activations is not None:
            i_activations = i_activations[..., tf.newaxis]
        for i in range(iterations):
            if False:
                leak = tf.zeros_like(b, optimize=True)
                leak = tf.reduce_sum(leak, axis=1, keep_dims=True)
                leaky_logits = tf.concat([leak, b], axis=1)
                leaky_routing = tf.nn.softmax(leaky_logits, dim=1)
                c = tf.split(leaky_routing, [1, output_capsule_num], axis=1)[1]
            else:
                c = tf.nn.softmax(b, 1)
                #        if i_activations is not None:
            #            tf.transpose(tf.transpose(c, perm=[0,2,1]) * i_activations, perm=[0,2,1])
            outputs = self.squash(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < iterations - 1:
                b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])
        poses = outputs
        activations = K.sqrt(K.sum(K.square(poses), 2))
        return poses, activations
