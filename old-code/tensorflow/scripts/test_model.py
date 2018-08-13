import tensorflow as tf

import pickle
import numpy as np
from tqdm import tqdm

from models.tfeat import TFeat
from datasets.ubc import UBCDataset

from scripts.eval_metrics import ErrorRateAt95Recall

# Define the algorithm flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 32,
                            """The size of the images to process""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """The number of channels in the images to process""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """The size of the mini-batch""")
tf.app.flags.DEFINE_string('train_name', 'notredame',
                          """The name of the dataset used to for training""")
tf.app.flags.DEFINE_string('test_name', 'liberty',
                          """The name of the dataset used to for training""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/patches_dataset',
                           """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('model_file', 'model',
                           """The filename of the model to evaluate""")

def run_evaluation():
    # load data
    dataset = UBCDataset(FLAGS.data_dir)
    dataset.load_by_name(FLAGS.test_name)

    # compute mean and std
    print('Loading training stats:')
    file = open('stats_%s.pkl' % FLAGS.train_name, 'r')
    mean, std = pickle.load(file)
    print('-- Mean: %s' % mean)
    print('-- Std:  %s' % std)
    
    # get patches
    patches = dataset._get_patches(FLAGS.test_name)
    matches = dataset._get_matches(FLAGS.test_name)

    # quick fix in order to have normalized data beforehand
    patches = preprocess_data(patches, mean, std)
    
    with tf.name_scope('inputs'):
        # User defined parameters
        BATCH_SIZE     = FLAGS.batch_size
        NUM_CHANNELS   = FLAGS.num_channels
        IMAGE_SIZE     = FLAGS.image_size

        # Define the input tensor shape
        tensor_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

        # Triplet place holders
        inputs1_pl  = tf.placeholder(
            dtype=tf.float32, shape=tensor_shape, name='inputs1_pl')
        inputs2_pl = tf.placeholder(
            dtype=tf.float32, shape=tensor_shape, name='inputs2_pl')

    # Creating the architecture
    tfeat_architecture = TFeat()
    tfeat_inputs1 = tfeat_architecture.model(inputs1_pl)
    tfeat_inputs2 = tfeat_architecture.model(inputs2_pl)
    
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # init the graph variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        # restore session from file
        saver = tf.train.import_meta_graph('%s.meta' % FLAGS.model_file)
        saver.restore(sess, FLAGS.model_file)

        offset = 0
        dists  = np.zeros(matches.shape[0],)
        labels = np.zeros(matches.shape[0],)

        for x in tqdm(xrange(matches.shape[0] // FLAGS.batch_size)):
            # get batch ids
            batch = matches[offset:offset + FLAGS.batch_size, :]

            # update the batch offset 
            offset += FLAGS.batch_size 

            # fetch the model with data
            descs1, descs2 = sess.run([tfeat_inputs1, tfeat_inputs2],
                feed_dict = {
                    inputs1_pl: patches[batch[:,0]],
                    inputs2_pl: patches[batch[:,1]]
            })

            # compute euclidean distances between descriptors
            for i in xrange(FLAGS.batch_size):
                idx = x * FLAGS.batch_size + i
                dists[idx]  = np.linalg.norm(descs1[i,:] - descs2[i,:])
                labels[idx] = batch[i,2]

        # compute the false positives rate
        fpr95 = ErrorRateAt95Recall(labels, dists)
        print 'FRP95: %s' % fpr95


def main(_):
    run_evaluation()


if __name__ == '__main__':
    tf.app.run()
