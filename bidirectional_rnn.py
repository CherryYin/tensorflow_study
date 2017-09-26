
# coding: utf-8

# In[16]:

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


# In[17]:

learning_rate = 0.01
max_samples = 400000
display_size = 10
batch_size = 128


# In[18]:

n_input = 28
n_step = 28
n_hidden = 256
n_class = 10


# In[19]:

x = tf.placeholder(tf.float32, shape=[None, n_step, n_input])
y = tf.placeholder(tf.float32, shape =[None, n_class])

Weight = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))   #参数共享力度比cnn还大
bias = tf.Variable(tf.random_normal([n_class]))


# In[20]:

def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])  #表示样本数量不固定
    x = tf.split(x, n_step)
    
    lstm_qx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_hx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx, lstm_hx, x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


# In[21]:

pred = BiRNN(x, Weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accurancy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


# In[22]:

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_step, n_input))
        sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
        if step % display_size == 0:
            acc = sess.run(accurancy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost, feed_dict = {x:batch_x, y:batch_y})
            print 'Iter' + str(step*batch_size) + ', Minibatch Loss= %.6f'%(loss) + ', Train Accurancy= %.5f'%(acc)
            #print loss, acc
        step += 1
    print "Optimizer Finished!"


# In[23]:

    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape(-1, n_step, n_input)
    test_label = mnist.test.labels[:test_len]
    print 'Testing Accurancy:%.5f'%(sess.run(accurancy, feed_dict={x: test_data, y:test_label}))


    Coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=Coord)




