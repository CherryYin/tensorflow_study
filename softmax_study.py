# coding:utf-8

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#创建新的会话，图的执行和保存一般是在会话中完成，但这里sess貌似没有用到，这点不太明白。
sess = tf.InteractiveSession()
#placeholder:占位符，提供输入数据的地方，不是tensor.一般只有x,y输入才需要用到。
#[None, 784]:占位符的shape，也是输入tensor的shape，取决于输入数据的shape。784表示列数，None几乎等于输入行数任意。
x = tf.placeholder(tf.float32, [None, 784])

#Variable：持久化保存tensor。tensor本身一用完就会消失，像w,b这种一直迭代的参数需要持久化存在。
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#tensorflow的nn中有大量神经网络组件，要用某种操作，首先到nn、train这样的模组种找。
#matmul矩阵乘法中的一个ops，最常用的ops有array、矩阵等文件下的ops，可在../tenserflow/python/ops下xx_ops.py中找。
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32,[None, 10])

#计算交叉熵，也就是求loss
#reduce_mean：对每个batch数据结果求均值
#reduce_sum：求和
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#GradientDescentOptimizer：随机梯度下降（SGD），0.5为其学习率，minimize的参数为最小化的目标tensor.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#初始化图中变量。
tf.global_variables_initializer().run()

#开始按Batch训练。
for i in range(1000):
    #读入100张图和对应的100个标签。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #执行优化
    #{x:batch_xs, y_:batch_ys}将数据输入到对应的placeholder中。
    train_step.run({x:batch_xs, y_:batch_ys})

#equal:判断两个参数是否相等，预测是否准确。
#argmax：求各预测数字中概率最大的一个。
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#计算搜索测试的平均准确率。
#cast：转换类型，将第一个参数的类型转化为第二参数指定的类型。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval({x:mnist.test.images, y_:mnist.test.labels})

