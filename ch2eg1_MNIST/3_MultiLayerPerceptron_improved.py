#  !/usr/bin/env python
#  -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf

tf.set_random_seed(0)
train_step_number = 10000
train_step_length = tf.placeholder(tf.float32) # decay
keep_ratio  = tf.placeholder(tf.float32) # donot set to a const. =1 on test dataset
min_learning_rate, max_learning_rate = 0.00003, 0.003
decay_speed = 2000.0

# 这里仍然使用多层感知机MLP建模识别，但加入使用ReLu、decay、dropout等。步骤：
# 1 描述模型
# 2 设置loss
# 3 优化方案
# - 最后，馈入数据训练即可。

# 模型 model
nn0, nn1, nn2, nn3, nn4, nn5, nn6 = 784, 400, 200, 100, 50, 20, 10

x = tf.placeholder(tf.float32, [None, nn0]) # 读入一批图片，存入这里；第一维是图片个数

W1  = tf.Variable(tf.truncated_normal([nn0, nn1], stddev=0.1))      # y1 = sigmoid(W1x + b1)
b1  = tf.Variable(tf.zeros([nn1])+0.1)           # W1, b1 同为模型参量
y1A = tf.nn.relu(tf.matmul(x, W1) + b1)          # 其他层类似
y1  = tf.nn.dropout(y1A, keep_ratio)

W2  = tf.Variable(tf.truncated_normal([nn1, nn2], stddev=0.1))
b2  = tf.Variable(tf.zeros([nn2])+0.1)# 用一个小的正数初始化偏置项，避免神元输出恒为0（dead）
y2A = tf.nn.relu(tf.matmul(y1, W2) + b2)
y2  = tf.nn.dropout(y2A, keep_ratio)

W3  = tf.Variable(tf.truncated_normal([nn2, nn3], stddev=0.1))
b3  = tf.Variable(tf.zeros([nn3])+0.1)
y3A = tf.nn.relu(tf.matmul(y2, W3) + b3)
y3  = tf.nn.dropout(y3A, keep_ratio)

W4  = tf.Variable(tf.truncated_normal([nn3, nn4], stddev=0.1))
b4  = tf.Variable(tf.zeros([nn4])+0.1)
y4A = tf.nn.relu(tf.matmul(y3, W4) + b4)
y4  = tf.nn.dropout(y4A, keep_ratio)

W5  = tf.Variable(tf.truncated_normal([nn4, nn5], stddev=0.1))
b5  = tf.Variable(tf.zeros([nn5])+0.1)
y5A = tf.nn.relu(tf.matmul(y4, W5) + b5)
y5  = tf.nn.dropout(y5A, keep_ratio)

W6 = tf.Variable(tf.truncated_normal([nn5, nn6], stddev=0.1))
b6 = tf.Variable(tf.zeros([nn6])+0.1)
yLogits = tf.matmul(y5, W6) + b6
y = tf.nn.relu(yLogits)

# loss
y_ = tf.placeholder(tf.float32, [None, 10]) # 读入与x相对应的标签

# 用交叉熵作为损失函数!!!
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yLogits, labels=y_))*100

# 自适应梯度下降优化
train_step = tf.train.AdamOptimizer(train_step_length).minimize(loss)

# data 准备数据
mnist = read_data_sets("MNIST_data", one_hot=True)
batch_size = 100  # 每批读入100个图片、标签

# 辅助op，用于检测模型精度
prediction_correction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))   # T,F,T...
accuracy = tf.reduce_mean(tf.cast(prediction_correction, tf.float32)) # 1,0,1...

# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # init vars
    sess.run(init)

    # training
    for i in range(train_step_number):
        this_step_length = min_learning_rate + \
                 (max_learning_rate-min_learning_rate) * math.exp(-i/decay_speed)
        #
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # load
        sess.run(train_step,
                feed_dict={x: batch_xs, y_: batch_ys,
                           train_step_length: this_step_length,
                           keep_ratio:0.75})
        #
        if (i % 1000) == 0:
            print(i,':\n', sess.run(W6), '\n', sess.run(b6), '\n')

    # desc - print result
    print("Accuarcy on test dataset: ",
        sess.run(accuracy,
                 feed_dict={x:mnist.test.images, y_:mnist.test.labels,
                            train_step_length: min_learning_rate,
                            keep_ratio:1}))

print("\nDone.")  # all done  ~0.982
