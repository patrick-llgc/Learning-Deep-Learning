"""
Defaults to 0, so all logs are shown. 
Set TF_CPP_MIN_LOG_LEVEL to 
    1 to filter out INFO logs, 
    2 to additionall filter out WARNING, 
    3 to additionally filter out ERROR.
"""
from __future__ import print_function
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print('INFO and WARNING turned off, ', end=" ")
print('TensorFlow version {}'.format(tf.__version__))

# tensors are const (values never change)
hello_const = tf.constant('Hello World!')

# session is an env for running a graph. It allocates operations to GPU/CPU
with tf.Session() as sess:
    output = sess.run(hello_const)
    print(output)

# feed_dict: Use the feed_dict parameter in tf.session.run() to set the placeholder tensor.
a = tf.placeholder(tf.string)
b = tf.placeholder(tf.int32)
c = tf.placeholder(tf.float32)
with tf.Session() as sess:
    output = sess.run(b, feed_dict={a: 'hi', b: 23, c: 32.0})
    print(output)

# Maths
# sub and mul has been removed in v1.0.1
add = tf.add(5, 2)
sub = tf.subtract(10, 4)
mul = tf.multiply(2, 5)
div = tf.div(10, 5)
with tf.Session() as sess:
    output = [sess.run(add), sess.run(sub), sess.run(mul), sess.run(div)]
    print(output)

# Variables, value can be changed, not tensor (const)
def variables():
    output = None
    x = tf.Variable([1, 2, 3, 4])
    # init = tf.initialize_all_variables()
    # initialize_all_variables deprecated
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(x)
    return output
print(variables())

# Logistic Regression
def logits():
    output = None
    x_data = [[1.0, 2.0], [2.5, 6.3]]
    test_weights = [[-0.3545495, -0.17928936], [-0.63093454, 0.74906588]]
    class_size = 2
    
    x = tf.placeholder(tf.float32)
    weights = tf.Variable(test_weights)
    biases = tf.Variable(tf.zeros([class_size]))
    
    # ToDo: Implement wx + b in TensorFlow
    logits = tf.matmul(weights, x)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(logits, feed_dict={x: x_data})
    return output
print(logits())
