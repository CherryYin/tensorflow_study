#coding:utf-8

from tutorials.image.cifar10 import cifar10_input
from tutorials.image.cifar10 import cifar10
import tensorflow as tf
import numpy as np
import time


max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
cifar10.maybe_download_and_extract()

def interface():
    mages_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
    images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)
    image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    label_holder = tf.placeholder(tf.int32, [batch_size])

    def variable_with_weight_loss(shape, stddev, w1):
        var = tf.Variable(tf.truncated_normal(shape, stddev))
        if w1 is not None:
            #weight_loss = tf.matmul(tf.nn.l2_loss(var), w1, name='weight_loss')
            weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return var

    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.05, w1=0.0)
    bias1 = tf.Variable(tf.constant(0.0, tf.float32, [64]))
    # conv_kernals_1 = tf.nn.conv2d(image_holder, weight1,stride=[1,1,1,1], padding='SAME')
    conv_kernals_1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
    # relu_1 = tf.nn.relu(tf.nn.add_bias(conv_kernals_1, bias1))
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv_kernals_1, bias1))
    pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=0.05, w1=0.0)
    bias2 = tf.Variable(tf.constant(0.1, tf.float32, shape=[64]))
    conv_kernels_2 = tf.nn.conv2d(norm_1, weight2, strides=[1, 1, 1, 1], padding='SAME')
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv_kernels_2, bias2))
    norm_2 = tf.nn.lrn(relu_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # -1是什么意思
    # dim取shape[1]是因为[0]是batch_size。
    reshape = tf.reshape(pool_2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.04)
    bias3 = tf.Variable(tf.constant(0.1, tf.float32, [384]))
    full_connect_1 = tf.matmul(reshape, weight3) + bias3
    relu_3 = tf.nn.relu(full_connect_1)

    weight4 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.04)
    bias4 = tf.Variable(tf.constant(0.1, tf.float32, shape=[192]))
    full_connect_2 = tf.matmul(relu_3, weight4) + bias4
    relu_4 = tf.nn.relu(full_connect_2)

    weight5 = variable_with_weight_loss([192, 10], stddev=1 / 192.0, w1=0.0)
    bias5 = tf.Variable(0.0, tf.float32, [10])
    logits = tf.matmul(relu_4, weight5) + bias5


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


