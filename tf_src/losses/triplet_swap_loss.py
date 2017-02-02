import tensorflow as tf


def compute_triplet_swap_loss(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope('triplet_swap_loss'):
        d_p_squared = tf.reduce_sum(tf.square(anchor_feature - positive_feature), 1)
        d_n_squared = tf.reduce_sum(tf.square(anchor_feature - negative_feature), 1)
        d_h_squared = tf.reduce_sum(tf.square(positive_feature - negative_feature), 1)

        d_star = tf.minimum(d_n_squared, d_h_squared)
        loss = tf.maximum(0., margin + d_p_squared - d_star)

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
