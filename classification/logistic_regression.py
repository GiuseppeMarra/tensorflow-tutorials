import tensorflow as tf
from sklearn import datasets
import matplotlib
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt





dataset_size = 100
input_size = 2
output_size = 1
d = 2
learning_rate = 0.0001

def act(x):

    # return tf.sigmoid
    mu = tf.Variable(initial_value=tf.truncated_normal([1,d]))
    return tf.exp(-tf.reduce_sum(tf.square(x - mu), axis=1))




# Blobs
X,Y = datasets.make_blobs(n_samples=dataset_size,n_features=2,centers=2, )

# Circles
# X, Y = datasets.make_circles(n_samples=100)

# Moons
# X,Y = datasets.make_moons(n_samples=dataset_size)


# # XOR
# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Y = np.array([0,1,1,0])
# dataset_size = 4


# Building The graph

x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
t = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

w = tf.Variable(initial_value=tf.truncated_normal([input_size,
                                                  output_size]))
b = tf.Variable(initial_value=tf.zeros([output_size]))

a = tf.matmul(x,w) + b
y = tf.sigmoid(a)

# Criterion
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = t,
                                        logits = a))

# Optimization
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

###Optimization/Running phase
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(3000):
     ex_x = np.reshape(X[i%dataset_size],[1,2])
     ex_y = np.reshape(Y[i%dataset_size], [1,1])
     feed_dict ={x: ex_x , t: ex_y}
     loss, _ = sess.run((cost,train_op), feed_dict )
     print(loss)


def decision_surface(y,sess, X):

    gridX = np.mgrid[np.min(X[:,0]):np.max(X[:,0]):100j, np.min(X[:,1]):np.max(X[:,1]):100j].reshape(2,-1).T
    gridY = np.reshape(sess.run(y, feed_dict={x: gridX}),[-1])
    return gridX, gridY


fig = plt.figure()

ax1 = fig.add_subplot(311)
ax1.plot(X[Y==0,0], X[Y==0,1], "og")
ax1.plot(X[Y==1,0], X[Y==1,1], "or")

ax3 = fig.add_subplot(312)
Y = np.reshape(sess.run(y, feed_dict={x: X}), [-1])
ax3.plot(X[Y<0.5,0], X[Y<0.5,1], "og")
ax3.plot(X[Y>=0.5,0], X[Y>=0.5,1], "or")

X,Y = decision_surface(y,sess, X)
ax2 = fig.add_subplot(313)
ax2.plot(X[Y<0.5,0], X[Y<0.5,1], "og")
ax2.plot(X[Y>=0.5,0], X[Y>=0.5,1], "or")



plt.show()






