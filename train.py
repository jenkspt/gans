from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True, reshape=False)
import skimage.io

import tensorflow as tf
import numpy as np
from model import GAN

batch_size = 50


gan = GAN(n_classes=0)

with tf.name_scope('global_step'):
    global_step = tf.Variable(0, trainable=False, name='value')
    _step = tf.assign(global_step, global_step+1, name='increment')

examples = tf.placeholder(tf.float32, [None, 28, 28, 1], 'examples')
labels = tf.placeholder(tf.float32, [None, 10], 'labels')
critic_fetch, gen_fetch = gan.build_train_graph(batch_size, examples)

summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('logdir', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    while True:
        try:
            for i in range(5):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                #batch_x = (batch_x - 128)/128
                feed = {examples:batch_x, labels:batch_y}
                _, critic_loss = sess.run(critic_fetch, feed_dict=feed)

            _, gen_loss = sess.run(gen_fetch)
            step = sess.run(_step)

            if step%100 == 0:
                summary = sess.run(summary_op, feed_dict=feed)
                summary_writer.add_summary(summary, step)
                print('Step {}, critic loss = {}, generator loss = {}'\
                        .format(step, critic_loss, gen_loss))
        except KeyboardInterrupt:
            break
    print('Exiting ...')
