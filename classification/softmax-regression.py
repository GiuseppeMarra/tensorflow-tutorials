from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf



"""PHASE 1: Data loading and preprocessing"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #Look at one_hot labels!!!!

"""PHASE 2: Graph/Model Building"""
# Building inputs to the graph: they are pairs inputs(i.e x), targets(i.e. y_)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#Summing up evidences
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
a = logits = tf.matmul(x, W) + b

#Transforming evidences into a probability distribution
y = tf.nn.softmax(a)


#Define objective function
cross_entropy= tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Define an optimization algorithm
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



"""PHASE 3: GRAPH RUNNING / LEARNING"""

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels})
print("Test accuracy: %.3f" % test_accuracy)




