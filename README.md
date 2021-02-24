# tage-python
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
x = tf.placeholder(shape=[None, 2], dtype=tf.float32)
import numpy as np

my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])
                     
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)


my_product = tf.multiply(x_data, m_const)

sess = tf.Session()

for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))
