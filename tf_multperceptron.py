
# coding: utf-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#输入层cell数量
in_units = 784
#隐含层数量，可以是自定义的，一般比输入层少。
h1_units = 300

#输入层W1的shape从[784, 10]改为[784,300],b1也从10改为300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
#dropout的保留百分比，用一个placeholder占位，使其可以自由配置。
#如果keep_prob被设为0.75,那么随机选择75%的节点信息有效，25%的节点的信息丢弃。
keep_prob = tf.placeholder(tf.float32)

#input->hidden采用relu激活。
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#input->hidden启用dropout
hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_dropout, W2) + b2)

y_=tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(3000):
    x_batches, y_batches = mnist.train.next_batch(100)
    #训练数据、标签、dropout百分比输入。
    train_step.run({x:x_batches, y_:y_batches, keep_prob:0.75})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#训练时，一般dropout百分比小于1,测试时，一般等于1.
print accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})


