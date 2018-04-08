# [MNIST](http://yann.lecun.com/exdb/mnist/)

The `hello, world` of `Machine Learning`.
See also [ML without a PhD](https://github.com/martin-gorner/tensorflow-mnist-tutorial)

- 60000个训练样本（mnist.train）
- 10000个测试样本（mnist.test）
- 每一个MNIST样本有两部分组成：

一张包含手写数字的图片和一个对应的标签。
    + 每一张图片包含28像素X28像素。
        - 图片存储为 28x28 = 784 维向量。
    + 像素的强度值，介于0和1之间。
    + 标签是介于0到9的数字，用来描述给定图片里表示的数字。
    + one-hot标签只有对应位的数字是1，其余各维都是0。
        - 比如，标签0将存储为([1,0,0,0,0,0,0,0,0,0,0])。
