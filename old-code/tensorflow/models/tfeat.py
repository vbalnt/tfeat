import tensorflow as tf

class TFeat(object):
    def __init__(self):
        """Create all the necessary variables for this CNN
        """
        # conv1
        with tf.variable_scope('conv1') as scope:
            self.kernel = self._variable_with_weight_decay('weights',
                                                           shape=[7, 7, 1, 32],
                                                           wd=1e-4)
            self.biases = self._variable_on_device('biases', [32],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        
        # conv2
        with tf.variable_scope('conv2') as scope:
            self.kernel = self._variable_with_weight_decay('weights',
                                                           shape=[6, 6, 32, 64],
                                                           wd=1e-4)
            self.biases = self._variable_on_device('biases', [64],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        
        # fc
        with tf.variable_scope('fc') as scope:
            self.weights = self._variable_with_weight_decay('weights',
                                                            shape=[8*8*64, 128],
                                                            wd=1e-4)
            self.biases = self._variable_on_device('biases', [128],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=True))

    def _variable_on_device(self, name, shape, initializer):
        """Helper to create a Variable stored on GPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device('gpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape,
                                  initializer=initializer, dtype=dtype)
        return var


    def _variable_with_weight_decay(self, name, shape, wd):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.
        Args:
            name: name of the variable
            shape: list of ints
            stddev: standard deviation of a truncated Gaussian
            wd: add L2Loss weight decay multiplied by this float. If None, weight
                decay is not added for this Variable.
        Returns:
            Variable Tensor
        """
        dtype = tf.float32
        var = self._variable_on_device(
            name,
            shape,
            tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(self, data):
        """The model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        with tf.variable_scope('conv1', reuse=True) as scope:
            conv = tf.nn.conv2d(data,
                                tf.get_variable('weights'),
                                strides=[1, 1, 1, 1],
                                padding='VALID')
            # Bias and tanh non-linearity.
            tanh = tf.nn.tanh(tf.nn.bias_add(conv, tf.get_variable('biases')))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
    	# the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(tanh,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='VALID')

        with tf.variable_scope('conv2', reuse=True) as scope:
            conv = tf.nn.conv2d(pool,
                                tf.get_variable('weights'),
                                strides=[1, 1, 1, 1],
                                padding='VALID')
            # Bias and tanh non-linearity.
            tanh = tf.nn.tanh(tf.nn.bias_add(conv, tf.get_variable('biases')))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        tanh_shape = tanh.get_shape().as_list()
        reshape = tf.reshape(tanh, [-1, 8*8*64])

	# Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        with tf.variable_scope('fc', reuse=True) as scope:
            matmul = tf.matmul(reshape, tf.get_variable('weights')) 
            fully  = tf.nn.tanh(matmul + tf.get_variable('biases'))
        return fully
