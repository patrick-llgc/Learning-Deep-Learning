# Learning-Deep-Learning
This repository contains my learning notes and the code snippets (in iPython Notebook or Python script format). The iPython notebook files are initially forked from [this repo](git@github.com:jessicayung/self-driving-car-nd.git) based on the Udacity course, but was heavily adopted with new notes and necessary changes to accommodate new API changes.

# Learning Notes on Deep Learning

This documents records the main takeaways during my learning of Deep Learning.

## Tensorflow
### Basic Concepts
- Tensorflow defines a graph (a collection of operations stacked in order) first, and then executes the operations on real data at run time. So when we perform operations in TF, we are [designing the architect](https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3) without running any calculation. The calculation happens inside a Session. So there are two stages in TF's code: the **Graph** level and the **Session** (evaluation) level. 
- Real data are not needed at definition of the graph. For this, inputs are represented by a datatype named `placeholder`. Only at run time, real data is fed into the `placeholder`s through a mapping dictionary `feed_dict`.
- `Variable`: Only variables keeps their data between multiple evaluation. All other tensors are temporary which means that they will be destroyed and inaccessible in your training for-loop without a proper feed_dict.- Nodes in Tensorflow's graph represents operation (op), and the edge represents the data (tensor) that flows between them.
- All data in Tensorflow are represented by a data type Tensor, with a static type, a rank and a shape. However its values have to be evaluated through tf.Session().run().
- Once a graph is defined, it has to be deployed with a session to get the output. A session is an environment that supports the execution of the operations. If two or more tensors needs to be evaluated, put them in a list and pass to run().



### Train a model
- Define a loss. A loss node has to be defined if we want to train the model. It is very common to use op like `tf.reduce_sum()` to sum across a certain axis.
	
	```
	pred = tf.nn.softmax() # prediction model
	label = tf.placeholder(tf.float32, [n_batch, n_class])
	loss = -tf.reduce_sum(label * tf.log(pred), axis=1) # cross entropy
	```

- Compute gradients. 

	```
	optimizer = tf.train.GradientDescentOptimizer(lr)
	train_step = optimizer.minimize(loss)
	```
- Train the model.

	```
	batch_x, batch_label = data.next_batch()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(train_step, feed_dict={x: batch_x, label: batch_label})
	```

### Breaking changes to Tensorflow's API (since early 2017)
Tensorflow has been evolving fast. When running code snippet online, especially those written in 2016, before many changes was implemented in API r1.0 in early 2017, we may encounter various syntax errors. See most updated list of API changes [here](https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/RELEASE.md). Among them, the most common ones are:

- `tf.multiply()` replaced `tf.mul()`
- `tf.subtract()` replaced `tf.sub()`
- `tf.negative()` replaced `tf.neg()`
- `tf.split` now takes argument in the order of `tf.split(value, num_or_size_splits, axis)`, reversing the previous orders for `axis` and `value`.
- `tf.global_variables_initializer` replaced `tf.initialize_all_variables`

### Tensor Shapes [link](http://stackoverflow.com/questions/37096225/how-to-understand-static-shape-and-dynamic-shape-in-tensorflow)
A tensor has a dynamic shape and a static (or inferred) shape, accessible by `tf.shape()` and `tf.Tensor.get_shape()` respectively. 

The static shape is a tuple or a list. The static shape is very useful to debug your code with print so you can check your tensors have the right shapes. The dynamic shape is itself a **tensor** describing the shape of the original tensor.

By default, a `placeholder` has a completely unconstrained shape, but you can constrain it by passing the optional shape argument.

```
w = tf.placeholder(tf.float32)                      # Unconstrained shape
x = tf.placeholder(tf.float32, shape=[None, None])  # Matrix of unconstrained size
y = tf.placeholder(tf.float32, shape=[None, 32])    # Matrix with 32 columns
z = tf.placeholder(tf.float32, shape=[128, 32])     # 128x32-element matrix
```

In general `shape=[None, 32]` is the most common way as to put some constraint to feature dimension but also be able to accommodate different batch sizes. 

In contrast, the learnable `Variable` generally has a known static shape. 
