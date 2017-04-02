# Learning-Deep-Learning
This repository contains my learning notes and the code snippets (in iPython Notebook or Python script format). 

# Learning Notes on Deep Learning

This documents records the main takeaways during my learning of Deep Learning.

## Tensorflow
### Basic Concepts
- Tensorflow defines a graph (a collection of operations stacked in order) first, and then executes the operations on real data at run time.
- Real data are not needed at definition of the graph. For this, inputs are represented by a datatype named `placeholder`. Only at run time, real data is fed into the `placeholder`s through a mapping dictionary `feed_dict`.
- Nodes in Tensorflow's graph represents operation (op), and the edge represents the data (tensor) that flows between them.
- All data in Tensorflow are represented by a data type Tensor, with a static type, a rank and a shape. However its values have to be evaluated through tf.Session().run().
- Once a graph is defined, it has to be deployed with a session to get the output. A session is an environment that supports the execution of the operations. If two or more tensors needs to be evaluated, put them in a list and pass to run().

###

### Breaking changes to Tensorflow's API
Tensorflow has been evolving fast. When running code snippet online, especially those written in 2016, before many changes was implemented in API r1.0 in early 2017, we may encounter various syntax errors. See most updated list of API changes [here](https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/RELEASE.md). Among them, the most common ones are:

- `tf.multiply()` replaced `tf.mul()`
- `tf.subtract()` replaced `tf.sub()`
- `tf.negative()` replaced `tf.neg()`
- `tf.split` now takes argument in the order of `tf.split(value, num_or_size_splits, axis)`, reversing the previous orders for `axis` and `value`.
- `tf.global_variables_initializer` replaced `tf.initialize_all_variables`