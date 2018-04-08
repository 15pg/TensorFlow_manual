#  !/usr/bin/env python
#  -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import tensorflow as tf

tf.set_random_seed(0)
train_step_number, train_step_length = 10000, 0.001

# 这里使用线性回归 (类似于 y=kx+b) 进行建模，以识别数字。步骤：：
# 1 描述模型
# 2 设置loss
# 3 优化方案
# - 如上有了模型，馈入数据训练即可。

# 模型 model
x = tf.placeholder(tf.float32, [None, 784]) # 读入一批图片，存入这里；第一维是图片个数
W = tf.Variable(tf.zeros([784, 10]))        # y = Wx + b
b = tf.Variable(tf.zeros([10]))             # W, b 同为模型参量
yLogits = tf.matmul(x, W) + b               # 描述模型，b向量将被广播到每一个样本!
y = tf.nn.softmax(yLogits)

# loss
y_ = tf.placeholder(tf.float32, [None, 10]) # 读入与x相对应的标签

#loss = -tf.reduce_mean(y_ * tf.log(y))*1e3  # 用交叉熵作为损失函数!!!
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yLogits, labels=y_))*100
# 按说loss乘不乘系数对梯度下降的方向选择没有影响，因此也不影响优化结果
# 但实践中，乘以系数会对预测精度产生明显的影响。大约是数值计算引入的问题，需细追究!!!!!!

# 梯度下降优化
train_step = tf.train.GradientDescentOptimizer(train_step_length).minimize(loss)

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
            print(i,':\n', sess.run(W), '\n', sess.run(b), '\n')

    # desc - print result
    print("Accuarcy on test dataset: ",
        sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

print("\nDone.")  # all done  ~0.925
