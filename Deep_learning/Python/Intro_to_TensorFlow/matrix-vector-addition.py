# Demonstration of xW+b
# Show that adding a 1x2 vector to a 2x2 matrix adds the 1x2 vector
# to each row of the 2x2 matrix

import tensorflow as tf

def run():
    output = None
    
    a = tf.add(tf.constant([[1,0],[0,1]]), tf.constant([2,3]))
    
    with tf.Session() as sess:
        output = sess.run(a)
    
    print(output)

run()
