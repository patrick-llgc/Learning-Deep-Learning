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

### Memory requirements imposed by conv layers
- An example taken from O'Reilly's *Hands-on machine learning with scikit-learn and tensorflow*: input 150x100 RGB image, one conv layer with 200 5x5 filters, 1x1 stride and `SAME` padding. The output parameters would be 200 feature maps of size 150x100, with a total number of parameters of (5x5x3+1)x200 = 15200 parameters. 
- Computation: Each of the 200 feature maps contains 150x100 neurons, and the each neuron needs to make weighted sum of 5x5x3 inputs, that is (5x5x3)x150x100x200=225 million multiplications. Including the same amount of addition, this requires 450 million flops.
- Storage: if each weight is stored in 32-bit float (double), then the output features takes 200x150x100x32/8~11.4 MB of RAM per instance. If a training batch contains 100 instance, the this layer would take up 1 GB of RAM!
- During training, every layer computed during the forward pass needs to be preserved for back-propagation, so the RAM needed is at least the total amount of RAM needed. 
- During inference, the RAM occupied by one layer can be released as soon as the next layer has been completed, so only as much RAM as required by two consecutive layers are needed.

### Pooling layers
- Pooling reduces the input image size and also makes the NN tolerate a bit more image shift (location invariance).
- Pooling works on every input channel independently. Generally you can pool over the height and width in each channel, or pool over the channels. You can not do both currently in tensorflow.

### CNN architecture
- Typical CNN architectures stack a few convolutional layers (each one followed by a ReLU layer) and a pooling layer. The image gets smaller and smaller but gets deeper and deeper as well.
- Common mistake is to make kernels too large. We can get the same effect as 9x9 kernels by stacking two 3x3 kernels. 
- Cross entropy cost function is preferred as it penalizes bad predictions much more, producing larger gradients and thus converging faster.

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
- For simple classification tasks, it generally is OK to **down-sample** the image first and then do the classification using CNN.
- Deep learning is usually good at bring up sensitivity, then afterwards use conventional computer vision or machine learning techniques to bring up specificity (filtering out false positives).


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

### Talks
#### Andrew Ng: Nuts and Bolts of Applying Deep Learning
- The video is available on [youtube](https://www.youtube.com/watch?v=F1ka6a13S9I&t=4279s).
- Decision making flowchart in tuning DL
![](images/Decision Making in Applying Deep Learning - Page 1.svg)
- Make sure dev and test are from the same distribution. 
	- Dev is the benchmark of tuning.
- Bias vs Variance tradeoff
	- In the era of deep learning, the bias and variance are not as closely coupled as traditional machine learning.
- Human level performance
	- It provides feasibility, data and insights.
	- Why improving after beating human level performance becomes harder? 
	- Human level performance gives guidance on which gap to focus on before hitting human level performance (e.g., are we doing good enough on training data error?).
	- Afterwards, it becomes unclear which area to focus on (e.g., hard to tell if is it a bias or variance problem).
	- What to do? Focus on sub-areas still lagging behind human level performance.
- Why it is increasingly important now? 
	- Since we are approaching human level performance now, and knowing human level performance is very useful information to drive decision making.
- How to define human level performance in order to drive algorithm development?
	- Given that human level performance are often used as proxy for theoretical optimal error rate (and measure the noise level of the data), it is most useful to get the best possible human level performance. 
	- Medical example of making diagnosis of a certain disease. Among the error rate results from a) average person 3%, b) average doctor 1%, c) expert doctor 0.8%, d) group of expert doctors 0.5%, d) is most useful. However, considering the difficulty of obtaining the labels, b) is most often used.
- What can AI/DL do? How can we put AI into our product?
	- Heuristic: Anything that a person can perform with < 1s of thought.
	- Predicting outcome of next in sequence of events.
- How to become a good DL researcher?
	- ML + DL basics
	- PhD student process: 
		- Read a lot of (20~50 at least) papers
		- replicate the result.
	- Dig into dirty work (but not only that)

### Recurrent Neural Networks (RNN)
- [link](https://www.youtube.com/watch?v=nFTQ7kHQWtc)
- Beautiful simplicity of backpropagation: every local gradient is a LOCAL worker in a GLOBAL chase for smaller loss function. 
- The gradient of sigmoid function $\frac{d\sigma(x)}{dx} = \sigma(x) (1-\sigma(x))$.
