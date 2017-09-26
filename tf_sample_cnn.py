
# coding: utf-8

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNSIT_data/", one_hot=True)
sess = tf.InteractiveSession()


# In[2]:
#由于W和b在各层中均要用到，先定义乘函数。
#tf.truncated_normal：截断正态分布，即限制范围的正态分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# In[7]:
#bias初始化值0.1.
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[12]:
#tf.nn.conv2d:二维的卷积
#conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,data_format=None, name=None)
#filter:A 4-D tensor of shape
#      `[filter_height, filter_width, in_channels, out_channels]`
#strides:步长，都是1表示所有点都不会被遗漏。1-D 4值，表示每歌dim的移动步长。
# padding:边界的处理方式，“SAME"、"VALID”可选
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#tf.nn.max_pool:最大值池化函数，即求2*2区域的最大值，保留最显著的特征。
#max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
#ksize:池化窗口的尺寸
#strides:[1,2,2,1]表示横竖方向步长为2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
#tf.reshape:tensor的变形函数。
#-1：样本数量不固定
#28,28：新形状的shape
#1:颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])


#卷积层包含三部分：卷积计算、激活、池化
#[5,5,1,32]表示卷积核的尺寸为5×5, 颜色通道为1, 有32个卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#经过2次2×2的池化后，图像的尺寸变为7×7,第二个卷积层有64个卷积核，生成64类特征，因此，卷积最后输出为7×7×64.
#tensor进入全连接层之前，先将64张二维图像变形为1维图像，便于计算。
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


#对全连接层做dropot
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)


#又一个全连接后foftmax分类
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
#AdamOptimizer：Adam优化函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#训练，并且每100个batch计算一次精度
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})


#在测试集上测试
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))