import tensorflow as tf


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow_src variables
    """
    with tf.name_scope('euclidean_distance):
		  d = tf.square(tf.sub(x, y))
		  d = tf.sqrt(tf.reduce_sum(d, 1))
    return d
