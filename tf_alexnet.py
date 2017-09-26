# coding: utf-8

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batches = 100

def print_activations(t):
    print t.op.name + ' '+ str(t.get_shape().as_list())


def variable_weight(shape):
    Weight = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1), name='weight') 
    return Weight


def variable_bias(shape, initial_value):
    bias = tf.Variable(tf.constant(initial_value, dtype=tf.float32, shape=shape), trainable=True, name='bias')
    return bias


def convAndReLU(images, shape, strides, scope_name):
    with tf.name_scope(scope_name) as scope:
        Weight = variable_weight(shape)
        bias = variable_bias([shape[3]], 0.0)
        conv = tf.nn.conv2d(images, Weight, strides=strides, padding='SAME')
        conv_and_bias = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(conv_and_bias, name='conv')
        print_activations(relu)
        parameters = [Weight, bias]
        
        return relu, parameters


def fc(input, shape, scope_name):
    with tf.name_scope(scope_name) as scope:
        Weight = variable_weight(shape)
        bias = variable_bias([shape[1]], 0.1)
        matmul = tf.matmul(input, Weight) + bias
        fc_layer = tf.nn.relu(matmul, name = 'relu')
        parameters = [Weight, bias]
        
    return fc_layer, parameters


def interface(images):
    parameters = []
    conv1, para1 = convAndReLU(images, [11, 11, 3, 96], [1, 4, 4, 1], 'conv1')
    parameters += para1
    lrn1 = tf.nn.lrn(conv1, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')
    print_activations(pool1)
    
    conv2, para2 = convAndReLU(pool1, [5, 5, 96, 256], [1, 1, 1, 1], 'conv2')
    parameters += para2
    lrn2 = tf.nn.lrn(conv2, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')
    print_activations(pool2)
    
    conv3, para3 = convAndReLU(pool2, [3, 3, 256, 384], [1, 1, 1, 1], 'conv3')
    conv4, para4 = convAndReLU(conv3, [3, 3, 384, 384], [1, 1, 1, 1], 'conv4')
    conv5, para5 = convAndReLU(conv4, [3, 3, 384, 256], [1, 1, 1, 1], 'conv5')
    parameters += para3
    parameters += para4
    parameters += para5
    pool3 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool3')
    print_activations(pool3)
    
    reshape = tf.reshape(pool3, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    
    fc1, para6 = fc(reshape, [dim, 4096], 'fc1')
    fc2, para7 = fc(fc1, [4096, 4096], 'fc2')
    fc3, para8 = fc(fc2, [4096, 1000], 'fc3')
    parameters += para6
    parameters += para7
    parameters += para8
    
    return fc3, parameters


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_square = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i%10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
        total_duration += duration
        total_duration_square += duration * duration
    mn = total_duration / num_batches
    print mn, total_duration_square
    vr = total_duration_square / num_batches - mn * mn
    print vr
    sd = math.sqrt(math.fabs(vr))
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], 
                                              dtype=tf.float32, stddev=0.1))
        fc3, parameters = interface(images)
        
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, fc3, 'Forward')
        objective = tf.nn.l2_loss(fc3)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Forward_backward')


run_benchmark()
