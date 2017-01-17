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

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random_normal([784, 128]))
w2 = tf.Variable(tf.random_normal([128, 64]))
w3 = tf.Variable(tf.random_normal([64, 128]))
w4 = tf.Variable(tf.random_normal([128, 784]))

encoder_l1 = tf.nn.sigmoid(tf.matmul(x, w1))
encoder_l2 = tf.nn.sigmoid(tf.matmul(encoder_l1, w2))
decoder_l1 = tf.nn.sigmoid(tf.matmul(encoder_l2, w3))
decoder_l2 = tf.nn.sigmoid(tf.matmul(decoder_l1, w4))

cost = tf.reduce_mean(tf.pow(x - decoder_l2, 2))
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
optimizer = tf.train.AdamOptimizer().minimize(cost)

training_epochs = 50
batch_size = 50

init_op = tf.initialize_all_variables()
session = tf.Session()
session.run(init_op)

total_batch = mnist.train.num_examples / batch_size
for epoch in xrange(training_epochs):
    for batch in xrange(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)	
        _, c = session.run([optimizer, cost], feed_dict = {x: batch_x})
    print("epoch=%d" % (epoch), "cost=", "{:.9f}".format(c))


# compare encoder with original image
examples_to_show = 10
encode_decode = session.run(decoder_l2, feed_dict={x: mnist.test.images[:examples_to_show]})
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.savefig("./image/originalVsDecoder.png")

source_embedding = session.run(encoder_l2, feed_dict={x: mnist.test.images[:examples_to_show]})
target_embedding = session.run(encoder_l2, feed_dict={x: mnist.test.images[examples_to_show:]})

topK = 4
f, a = plt.subplots(topK + 1, 10, figsize=(10, topK + 1))
for i in xrange(examples_to_show):
    min_idx = np.argsort(np.sum((target_embedding - source_embedding[i]) ** 2, axis = 1))
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))           
    for j in xrange(topK):
        a[j + 1][i].imshow(np.reshape(mnist.test.images[examples_to_show + min_idx[j]], (28, 28)))

f.savefig("./image/closestImage.png")
        
