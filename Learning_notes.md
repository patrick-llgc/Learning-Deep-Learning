# Learning Notes on Deep Learning

This documents records the main takeaways during my learning of Deep Learning.

## Tensorflow
### Basic Concepts
- Tensorflow defines a graph (a collection of operations stacked in order) first, and then executes the operations on real data at run time. So when we perform operations in TF, we are [designing the architect](https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3) without running any calculation. The calculation happens inside a Session. So there are two stages in TF's code: the **Graph** level and the **Session** (evaluation) level. 
- Real data are not needed at definition of the graph. For this, inputs are represented by a datatype named `placeholder`. Only at run time, real data is fed into the `placeholder`s through a mapping dictionary `feed_dict`.
- `Variable`: a class. The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. The initial value defines the type and shape of the variable. Only variables keeps their data between multiple evaluation (across calls to `run()`). All other tensors are temporary which means that they will be destroyed and inaccessible in your training for-loop without a proper feed_dict. Variables have to be explicitly initialized before you can run Ops that use their value.
- Nodes in Tensorflow's graph represents operation (op), and the edge represents the data (tensor) that flows between them.
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

### Using GPU
Use the following code to list all available devices:

```
import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

with tf.device('/cpu:1'):
	# creates a graph
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print sess.run(c)
```
The output may be produced on the console from where you ran the Jupyter Notebook. 

### Broadcasting rules
Tensorflow's broadcasting rules are designed to follow those of `numpy`'s.
When operating on two arrays, NumPy compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are compatible when

- they are equal, or
- one of them is 1

### Random seed (graph level and op level)
`tf.set_random_seed(seed)`'s interactions with operation-level seeds is as follows: [docstring](https://www.tensorflow.org/api_docs/python/tf/set_random_seed)

- If neither the graph-level nor the operation seed is set: A random seed is used for this op.
- If the graph-level seed is set, but the operation seed is not: The system deterministically picks an operation seed in conjunction with the graph-level seed so that it gets a unique random sequence.
- If the graph-level seed is not set, but the operation seed is set: A default graph-level seed and the specified operation seed are used to determine the random sequence.
- If both the graph-level and the operation seed are set: Both seeds are used in conjunction to determine the random sequence.

### Notes on Sherry Moore's tensorflow tutorial
- Youtube Video [Link](https://www.youtube.com/watch?v=Ejec3ID_h0w&t=2810s)
- Variables have to be explicitly initialized.

```
a = tf.Variable(tf.random_uniform([1], 0, 1))
sess = tf.Session()
sess.run(a.initializer)
print(sess.run(a))
```
```
b = tf.random_normal([1], 0, 1)
print(sess.run(b))
```
- There is no `tf.sum()` method. Instead, `tf.reduce_sum(keep_dims=False)` is very similar to `np.sum(keepdims=False)`. Pay special attention to the broadcasting rule.
- `np.random.normal()` **does not** take any `dtype` argument. It has to be explicitly defined as, for example, `np.random.normal().astype(np.float32)`.
- Best practice of `import`: 

```
from package.subpackage1.subpackage2 import subpackage3
subpackage3.name
```
- placeholder (NxD) Hidden layer 1 weight (DxH1) Hidden layer 2 weight (H1xH2). y = X * W + b


### Resources:
- Debugging tips in TF [link](https://wookayin.github.io/tensorflow-talk-debugging/#9)
- Coursera course by Hinton [link](https://www.coursera.org/learn/neural-networks/home/week/12)
- Kaggle Data Bowl [link](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook)
- Blogs
	- [lab41](https://gab41.lab41.org/lab41-reading-group-deep-networks-with-stochastic-depth-564321956729)
	- [codesachin](https://codesachin.wordpress.com/2017/02/19/residual-neural-networks-as-ensembles/)
