import os
import sys
import PIL
import PIL.Image
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x_with_noise = tf.placeholder(tf.float32, shape=[None, 784])
x_without_noise = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random_normal([784, 784]))
#w2 = tf.Variable(tf.random_normal([256, 256]))
#w3 = tf.Variable(tf.random_normal([256, 256]))
w4 = tf.Variable(tf.random_normal([784, 784]))

encoder_l1 = tf.nn.sigmoid(tf.matmul(x_with_noise, w1))
#encoder_l2 = tf.nn.sigmoid(tf.matmul(encoder_l1, w2))
decoder_l1 = tf.nn.sigmoid(tf.matmul(encoder_l1, w4))
#decoder_l1 = tf.nn.sigmoid(tf.matmul(encoder_l2, w3))
#decoder_l2 = tf.nn.sigmoid(tf.matmul(decoder_l1, w4))

cost = tf.reduce_mean(tf.pow(x_without_noise - decoder_l1, 2))
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)

training_epochs = 20
batch_size = 50
noise_ratio = 0.3

init_op = tf.initialize_all_variables()
session = tf.Session()
session.run(init_op)

total_batch = mnist.train.num_examples / batch_size
for epoch in xrange(training_epochs):
    for batch in xrange(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        x_with_noise_batch = batch_x + noise_ratio * np.abs(np.random.normal(size=batch_x.shape))
        _, c = session.run([optimizer, cost], feed_dict = {x_without_noise: batch_x, x_with_noise: x_with_noise_batch})
    print("epoch=%d" % (epoch), "cost=", "{:.9f}".format(c))


# compare encoder with original image
examples_to_show = 10
x_with_noise_batch = mnist.test.images[:examples_to_show] + \
                     noise_ratio * np.abs(np.random.normal(size=mnist.test.images[:examples_to_show].shape))
encode_decode = session.run(decoder_l1, feed_dict={x_with_noise: x_with_noise_batch})

# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
plt.gray()
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(x_with_noise_batch[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.savefig("./image/originalVsNoiseV2.png")
