# nn.py

"""
This script builds and runs a graph with miniflow.
Q1: x + y + z
Q2: (x + y) + y
Q3: x*w_x + y*w_y + z*w_z + bias

"""

from miniflow import *

# Calculate x + y + z
# declare x, y, z to be input neurons
x, y, z = Input(), Input(), Input()
f = Add(x, y, z)
feed_dict = {x: 10, y: 5, z: 3}
sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)
print("{} + {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], output))


# Calculate (x + y) + y
# the output neuron can be as constructed to a complex level
# but the feed_dict should only contain the input neurons
f3 = Add(Add(x, y), y)
sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f3, sorted_neurons)
print("({} + {}) + {} = {} (according to miniflow)".format(
    feed_dict[x], feed_dict[y], feed_dict[y], output))

# Calculate x*w_x + y*w_y + z*w_z + bias
inputs = [x, y, z]
weight_x, weight_y, weight_z = Input(), Input(), Input()
weights = [weight_x, weight_y, weight_z]

bias = Input()
f = Linear(inputs, weights, bias)

feed_dict = {
    x: 6,
    y: 14,
    z: 3,
    weight_x: 0.5,
    weight_y: 0.25,
    weight_z: 1.4,
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
print("Results of linear combination is {} (according to miniflow)".format(output))