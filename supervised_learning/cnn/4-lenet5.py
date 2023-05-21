#!/usr/bin/env python3
"""Module convolutional neural netwwork"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    """
    w_init = tf.contrib.layers.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=w_init)
    opt_1 = C1(x)

    P1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    opt2 = P1(opt_1)

    C2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=w_init)

    opt3 = C2(opt2)

    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    opt4 = P2(opt3)

    opt4_2 = tf.layers.Flatten()(opt4)
    FC3 = tf.layers.Dense(activation=tf.nn.relu,
                          units=120,
                          kernel_initializer=w_init)
    opt5 = FC3(opt4_2)

    FC4 = tf.layers.Dense(activation=tf.nn.relu,
                          units=84,
                          kernel_initializer=w_init)
    opt6 = FC4(opt5)

    FC5 = tf.layers.Dense(units=10,
                          kernel_initializer=w_init)
    opt7 = FC5(opt6)

    softmax = tf.nn.softmax(opt7)

    loss = tf.losses.softmax_cross_entropy(y, logits=opt7)

    y_pred = tf.math.argmax(opt7, axis=1)
    y = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(equality, "float"))

    train = tf.train.AdamOptimizer().minimize(loss)

    return softmax, train, loss, accuracy
