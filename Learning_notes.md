# Learning Notes on Deep Learning

This documents records the main takeaways during my learning of Deep Learning.

## Tensorflow

### FAQ
First read the [FAQ](https://www.tensorflow.org/programmers_guide/faq#running_a_tensorflow_computation)!

### Basic Concepts
- Tensorflow defines a graph (a collection of operations stacked in order) first, and then executes the operations on real data at run time. So when we perform operations in TF, we are [designing the architect](https://blog.metaflow.fr/tensorflow-a-primer-4b3fa0978be3) without running any calculation. The calculation happens inside a Session. So there are two stages in TF's code: the **Graph** level and the **Session** (evaluation) level. 
- Real data are not needed at definition of the graph. For this, inputs are represented by a datatype named `placeholder`. Only at run time, real data is fed into the `placeholder`s through a mapping dictionary `feed_dict`.
- `Variable`: a class. The Variable() constructor requires an initial value for the variable, which can be a Tensor of any type and shape. The initial value defines the type and shape of the variable. Only variables keeps their data between multiple evaluation (across calls to `run()`). All other tensors are temporary which means that they will be destroyed and inaccessible in your training for-loop without a proper feed_dict. Variables have to be explicitly initialized before you can run Ops that use their value.
- Nodes in Tensorflow's graph represents operation (op), and the edge represents the data (tensor) that flows between them.
- All data in Tensorflow are represented by a data type Tensor, with a static type, a rank and a shape. However its values have to be evaluated through tf.Session().run().
- Once a graph is defined, it has to be deployed with a session to get the output. A session is an environment that supports the execution of the operations. If two or more tensors needs to be evaluated, put them in a list and pass to run().
- The role of Python code is therefore to build this external computation graph, and to dictate which parts of the computation graph should be run.
- In the TensorFlow system, tensors are described by a unit of dimensionality known as **rank**. Tensor rank is **NOT** the same as **matrix rank**.

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

- difference between `tf.shape()` and `Tensor.get_shape()`

```
a = tf.random_normal([2, 5, 4])
print(a.get_shape())  # ==> (2, 5, 4)
print(sess.run(tf.shape(a)))  # ==> [2 5 4]
```

- Static shape `Tensor.get_shape()` is evaluated at graph construction time, while dynamic shape `tf.shape(Tensor)` is evaluated at runtime. 
- **N.B.** Static (inferred) shape may be incomplete. Eval dynamic shape in a session. 

	> In TensorFlow, a tensor has both a static (inferred) shape and a dynamic (true) shape. The static shape can be read using the tf.Tensor.get_shape method: this shape is inferred from the operations that were used to create the tensor, and may be partially complete. If the static shape is not fully defined, the dynamic shape of a Tensor t can be determined by evaluating tf.shape(t).
	
### x.set_shape() vs x = tf.reshape(x)
`set_shape` specifies undefined information, no copy involved. tf.reshape does a **shallow** copy (not expensive either). [link](http://stackoverflow.com/questions/35451948/clarification-on-tf-tensor-set-shape)

```
a = tf.placeholder(tf.float32, (None, 10))
print('{} initial'.format(a.get_shape()))
a.set_shape((5, 10))
print('{} after set shape'.format(a.get_shape()))
try:
    a.set_shape((1, 5, 10))
except:
    print('cannot set_shape to (1, 5, 10) shape incompatible!')
a = tf.reshape(a, [1, 5,-1])
print('{} reshaped (copied)'.format(a.get_shape()))

# ==>
"""
(?, 10) initial
(5, 10) after set shape
cannot set_shape to (1, 5, 10) shape incompatible!
(1, 5, 10) reshaped (copied)
"""
```

### NCHW (faster on cuDNN) vs NHWC (faster on CPU)
The current recommendation is that users support both formats in their models. In the long term, we plan to rewrite graphs to make switching between the formats transparent. [link](https://www.tensorflow.org/performance/performance_guide#use_nchw_image_data_format)

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
- Youtube Video [Link](https://www.youtube.com/watch?v=Ejec3ID_h0w&t=2810s), Detailed [documentation](https://www.tensorflow.org/get_started/mnist/mechanics)
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
- `sparse_softmax_cross_entropy_with_logits` and `softmax_cross_entropy_with_logits` [link](http://stackoverflow.com/questions/37312421/tensorflow-whats-the-difference-between-sparse-softmax-cross-entropy-with-logi)

### Epochs vs steps
An epoch usually means one iteration over all of the training data. For instance if you have 20,000 images and a batch size of 100 then the epoch should contain 20,000 / 100 = 200 steps. [link](http://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs)

### Saving models for re-use
- blog [link](https://nathanbrixius.wordpress.com/2016/05/24/checkpointing-and-reusing-tensorflow-models/)
- A typical scenario has three steps:
	1. Creating a Saver and telling the Saver which variables you want to save,
	2. Save the variables to a file,
	3. Restore the variables from a file when they are needed.
- `tf.train.import_meta_graph`[link](https://www.tensorflow.org/api_docs/python/tf/train/import_meta_graph)
- Checkpoint file only saves the weights, and the graph itself can be recovered from the meta file. ([link](http://stackoverflow.com/questions/36195454/what-is-the-tensorflow-checkpoint-meta-file)) There are two parts to the model, the model definition, saved by Supervisor as graph.pbtxt in the model directory and the numerical values of tensors, saved into checkpoint files like model.ckpt-1003418. ([link2](http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model))
- How to save model [link](http://stackoverflow.com/documentation/tensorflow/5000/save-and-restore-a-model-in-tensorflow#t=201704251751077840717)
 
	> Note that in this example, while the saver actually saves both the current values of the variables as a checkpoint and the structure of the graph (*.meta), no specific care was taken w.r.t how to retrieve e.g. the placeholders x and y once the model was restored. E.g. if the restoring is done anywhere else than this training script, it can be cumbersome to retrieve x and y from the restored graph (especially in more complicated models). To avoid that, always give names to your variables / placeholders / ops or think about using tf.collections as shown in one of the remarks.

### Reading data
Three ways to load data [link](https://www.tensorflow.org/programmers_guide/reading_data)

1. Feeding: using `feed_dict` when running each step.

	A placeholder exists solely to serve as the target of feeds. It is not initialized and contains no data. A placeholder generates an error if it is executed without a feed, so you won't forget to feed it.

	This is the easiest way, but parsing could be a bottleneck. In that case, build **input pipelines**. 
	
	Unless for a special circumstance or for example code, **DO NOT** feed data into the session from Python variables, e.g. dictionary. ([link](https://www.tensorflow.org/performance/performance_guide))

	```
	# This will result in poor performance.
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	```

2. Reading from files: input pipeline read from file at the beginning of a TF graph
3. Preloaded data: a constant or Variable in the graph holds all the data (for small datasets)



### A minimal example of saving and restoring a model
```
import tensorflow as tf
tf.reset_default_graph()
w1 = tf.Variable(tf.truncated_normal([10]), name='w1')
tf.add_to_collection('weights', w1)
saver = tf.train.Saver()

# save graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, r'/tmp/mnist/mymodel', global_step=10)
```
From a different station

```
import tensorflow as tf
# load graph (even from a different station)
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph(r'/tmp/mnist/mymodel-10.meta')
    new_saver.restore(sess,r'/tmp/mnist/mymodel-10')
    new_weights = tf.get_collection('weights')[0]
    print(sess.run(new_weights))
```
Another example

```
import tensorflow as tf

def save(checkpoint_file='/tmp/mnist/hello.chk'):
    with tf.Session() as session:
        x = tf.Variable([42.0, 42.1, 42.3], name='x')
        y = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='y')
        not_saved = tf.Variable([-1, -2], name='not_saved')
        session.run(tf.global_variables_initializer())

        print(session.run(tf.global_variables()))
	    # saver = tf.train.Saver([x, y])
        saver = tf.train.Saver()
        saver.save(session, checkpoint_file)

def restore(checkpoint_file='/tmp/mnist/hello.chk'):
    x = tf.Variable(-1.0, validate_shape=False, name='x')
    y = tf.Variable(-1.0, validate_shape=False, name='y')
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file)
        print(session.run(tf.global_variables()))

def restore2(checkpoint_file='/tmp/mnist/hello.chk'):
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
        saver.restore(session, checkpoint_file)
        session.run(tf.global_variables_initializer()) #needed
        print(session.run(tf.global_variables()))
        
def reset():
    tf.reset_default_graph() # destroys the graph
    
save() # saves [x, y, not_saved]
reset()
restore() # loads [x, y]
reset()
restore2() # loads [x, y, not_saved]
```

### Trainable Variables
tf.trainable_variables() returns a list of all trainable variables. tf.Variable(trainable=False) will not add variables to this list.

### Sharing variables
- [TF how-to](https://www.tensorflow.org/programmers_guide/variable_scope)
- iPython [Notebook](./ipynb/sherrym_tf_tutorial/Sharing_variables.ipynb)
- `v = tf.get_variable(name, shape, dtype, initializer)` retrieves variable, and creates one if not existed yet.
1. Case 1: the scope is set for creating new variables, as evidenced by tf.get_variable_scope().reuse == False.
2. Case 2: the scope is set for reusing variables, as evidenced by tf.get_variable_scope().reuse == True.
- `tf.variable_scope()` carries a prefix name and a reuse flag. `reuse` parameter is inherited in all sub-scopes.
- name scope vs variable scope: name scope is ignored by tf.get_variable().
### Questions：

1. Why do we have to normalize the stddev parameter during initialization?

	> One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". [link](https://www.tensorflow.org/get_started/mnist/pros)
2. What does `import_meta_graph` and `restore` do?

	> `import_meta_graph` loads the graph from the meta file, and `restore` recovers the weights of the trainable variables.
	
3. What does `add_to_collection` and `get_collection` do?
	> Makes it easier to retrieve variables from restored graph
4. How to use [input pipeline](http://web.stanford.edu/class/cs20si/lectures/notes_09.pdf)?

### Tips:
- Remember to use `tf.reset_default_graph()` to clear before training, especially during interactive development.
- Use fused batch norm in DNN.
- Use placeholder to hold the dropout probability during training and evaluation.
- Adam uses a adaptive learning rate.
- Use xentropy during training and accuracy during evaluation

```
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```


### Running on multiple devices
- https://www.tensorflow.org/tutorials/using_gpu
- https://www.tensorflow.org/tutorials/deep_cnn
- Using GPU [guide](http://learningtensorflow.com/lesson10/)
- Buying [guide](http://timdettmers.com/2017/04/09/which-gpu-for-deep-learning/#tldr)
- Guide Lines:
	1. FP32 runs much faster than FP64 on GPU. Theoretically FP32 should be twice as fast as FP64, but most GPU's don't have as many FP64 units and FP32 operations can be 32 times faster than FP64. This ratio depends on GPU architecture. However if the operation is memory bounded such as matrix transposition, this ideal number can be achieved. ([link](http://arrayfire.com/explaining-fp64-performance-on-gpus/)) 
	2. Use FP32 as default for float point calculations. It has enough precision for most deep learning cases.

### Resources:
- Cross entropy: A great visual [guide](http://colah.github.io/posts/2015-09-Visual-Information/) to cross entropy
- Visualizing graph using tensorboard[tutorial](http://learningtensorflow.com/Visualisation/)
- Debugging tips in TF [link](https://wookayin.github.io/tensorflow-talk-debugging/#9)
- Coursera course by Hinton [link](https://www.coursera.org/learn/neural-networks/home/week/12)
- Kaggle Data Bowl [link](https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial/notebook)
- Blogs
	- [lab41](https://gab41.lab41.org/lab41-reading-group-deep-networks-with-stochastic-depth-564321956729)
	- [codesachin](https://codesachin.wordpress.com/2017/02/19/residual-neural-networks-as-ensembles/)
> I've read a lot of research papers (DeepMind, Google Brain, Facebook, NYU, Stanford, etc.), blogs (Nervana Systems, Indico, Colah, Otoro's Blog, etc.), lecture notes (Stanford cs231n, cs224d, cs229), and tutorials (Quoc Le's tutorial, TensorFlow, etc.), and have watched a lot of videos (Hugo Larochelle's tutorials, Stanford cs229, TedTalks, lectures by Yann LeCun <3, etc.) [link](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=pd_sim_14_1?_encoding=UTF8&pd_rd_i=0262035618&pd_rd_r=ZB5XTPQHEMADR1H4VZ5Q&pd_rd_w=b08ty&pd_rd_wg=kNTBU&psc=1&refRID=ZB5XTPQHEMADR1H4VZ5Q)
