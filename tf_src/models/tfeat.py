import tensorflow as tf


class TFeat(object):
    def __init__(self, wd, activation='tanh'):
        """Create all the necessary variables for this CNN
        """
        self.wd = wd
        self.activation = activation
        self.initialization = tf.contrib.layers.xavier_initializer_conv2d(uniform=True)

        # conv1
        with tf.variable_scope('conv1'):
            self.kernel = self._variable_with_weight_decay(
                'weights', shape=[7, 7, 1, 32], wd=self.wd)

            self.biases = self._variable_on_device(
                'biases', [32], self.initialization)
        
        # conv2
        with tf.variable_scope('conv2'):
            self.kernel = self._variable_with_weight_decay(
                'weights', shape=[6, 6, 32, 64], wd=self.wd)

            self.biases = self._variable_on_device(
                'biases', [64], self.initialization)
        
        # fc1
        with tf.variable_scope('fc1'):
            self.weights = self._variable_with_weight_decay(
                'weights', shape=[8*8*64, 128], wd=self.wd)

            self.biases = self._variable_on_device(
                'biases', [128], self.initialization)

    def _variable_on_device(self, name, shape, initializer):
        """Helper to create a Variable stored on GPU memory.
        Args:
            name: name of the variable
            shape: list of ints
            initializer: initializer for Variable
        Returns:
            Variable Tensor
        """
        with tf.device('/gpu'):
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
        var = self._variable_on_device(name, shape, self.initialization)
        if wd is not None:
            if tf.__version__ == '0.11.0':
                weight_decay = tf.mul(tf.nn.l2_loss(var), wd,
                                      name='weight_loss')
            else:
                weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                           name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def h(self, x):
        """Apply a non-linear activation to the given input"""
        if self.activation == 'tanh':
            out = tf.nn.tanh(x)
        elif self.activation == 'relu':
            out = tf.nn.relu(x)
        elif self.activation == 'relu6':
            out = tf.nn.relu6(x)
        else:
            raise Exception('Not supported activation: {}'
                            .format(self.activation))
        return out

    def conv_block(self, name, x):
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        with tf.variable_scope(name, reuse=True):
            out = tf.nn.conv2d(x,
                               tf.get_variable('weights'),
                               strides=[1, 1, 1, 1],
                               padding='VALID')
            # Bias and non-linearity.
            out = self.h(tf.nn.bias_add(out, tf.get_variable('biases')))
        return out

    def linear_block(self, name, x):
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        with tf.variable_scope(name, reuse=True):
            out = tf.matmul(x, tf.get_variable('weights'))
            out = self.h(out + tf.get_variable('biases'))
        return out

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(self, x):
        # first convolution
        out = self.conv_block('conv1', x)

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        out = tf.nn.max_pool(out,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID')

        # second convolution
        out = self.conv_block('conv2', out)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        out = tf.reshape(out, [-1, 8*8*64])

        # linear layer
        out = self.linear_block('fc1', out)
        return out
