import tensorflow as tf


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    """
    Compute the contrastive loss as in

    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m

    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin

    **Returns**
     Return the loss operation
    """

    with tf.name_scope('triplet_loss'):
        d_p_squared = tf.reduce_sum(tf.square(anchor_feature - positive_feature), 1)
        d_n_squared = tf.reduce_sum(tf.square(anchor_feature - negative_feature), 1)

        loss = tf.maximum(0., margin + d_p_squared - d_n_squared)

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
