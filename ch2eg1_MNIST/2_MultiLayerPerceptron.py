#  !/usr/bin/env python
#  -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf

tf.set_random_seed(0)
train_step_number, train_step_length = 10000, 0.001

# 这里使用多层感知机MLP进行建模，以识别数字。步骤都一样的：
# 1 描述模型
# 2 设置loss
# 3 优化方案
# - 最后，馈入数据训练即可。

# 模型 model
nn0, nn1, nn2, nn3 = 784, 200, 50, 10 # nn0输入到第一层，第一层的输出为200神元，第三层的输出为10神元

x = tf.placeholder(tf.float32, [None, nn0]) # 读入一批图片，存入这里；第一维是图片个数

W1 = tf.Variable(1e-3*tf.random_normal([nn0, nn1]))      # y1 = sigmoid(W1x + b1)
b1 = tf.Variable(1e-3*tf.random_normal([nn1]))           # W1, b1 同为模型参量
y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)   # 其他层类似

W2 = tf.Variable(1e-3*tf.random_normal([nn1, nn2]))
b2 = tf.Variable(1e-3*tf.random_normal([nn2]))
y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2)

W3 = tf.Variable(1e-3*tf.random_normal([nn2, nn3]))
b3 = tf.Variable(1e-3*tf.random_normal([nn3]))
yLogits = tf.matmul(y2, W3) + b3            # 描述模型，b向量将被广播到每一个样本!
y = tf.nn.softmax(yLogits)

# loss
y_ = tf.placeholder(tf.float32, [None, 10]) # 读入与x相对应的标签

# 用交叉熵作为损失函数!!!
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yLogits, labels=y_))*100

# 自适应梯度下降优化
# 注意，与线性回归的优化方法不同
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
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # load
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if (i % 1000) == 0:
            print(i,':\n', sess.run(W3), '\n', sess.run(b3), '\n')

    # desc - print result
    print("Accuarcy on test dataset: ",
        sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

print("\nDone.")  # all done  ~0.980
