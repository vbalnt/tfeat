import tensorflow as tf
import numpy as np

import os
import cv2  # imported here just to set the random seed

from models.tfeat import TFeat
from datasets.ubc import UBCDataset
from losses.triplet_loss import compute_triplet_loss

from scripts.process import run_model, eval_model
from scripts.sampling import generate_triplets
from scripts.preprocess import normalize_data, load_stats

# Define the algorithm flags

FLAGS = tf.app.flags.FLAGS

# network parameters
tf.app.flags.DEFINE_integer('image_size', 32,
                            """The size of the images to process""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """The number of channels in the images to process""")
tf.app.flags.DEFINE_string('activation', 'tanh',
                           """The default activation function during the training""")
# training parameters
tf.app.flags.DEFINE_string('train_name', 'notredame',
                           """The default dataset name for training""")
tf.app.flags.DEFINE_string('test_name', 'liberty',
                           """The default dataset name for testing""")
tf.app.flags.DEFINE_integer('num_triplets', 128,
                            """The default number of pairs to generate""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """The size of the mini-batch""")
tf.app.flags.DEFINE_integer('num_epochs', 60,
                            """The number of iterations during the training""")
tf.app.flags.DEFINE_float('margin', 1.0,
                          """The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          """The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_float('weight_decay', 1e-4,
                          """weight decay (default: 1e-4)""")
tf.app.flags.DEFINE_string('optimizer', 'momentum',
                           """The optimizer to use (default: SGD)""")
# logging stuff
tf.app.flags.DEFINE_string('data_dir', '/tmp/patches_dataset',
                           """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('log_dir', '/tmp/tensorboard_log',
                           """The default path to the logs directory""")
tf.app.flags.DEFINE_string('log_name', 'triplet_toy',
                           """The default name for logging""")
# device parameters
tf.app.flags.DEFINE_string('gpu_id', '1',
                           """The default GPU id to use""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """random seed (default: 666)""")

# to use only a single gpu
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.log_name)

# set the seed for all the random stuff
np.random.seed(FLAGS.seed)
cv2.setRNGSeed(FLAGS.seed)

# log paramters to a file
with open('flags.txt', 'w') as text_file:
    text_file.write(str(FLAGS.__flags))


def run_training():
    # load data
    dataset = UBCDataset(FLAGS.data_dir, test=False)
    dataset.load_by_name(FLAGS.train_name)
    dataset.load_by_name(FLAGS.test_name)

    # Load mean and std
    print('Loading training stats:')
    mean, std = load_stats(FLAGS.train_name)
    print('-- Mean: %s' % mean)
    print('-- Std:  %s' % std)
    
    # get patches
    patches_train = dataset.get_patches(FLAGS.train_name)
    patches_test = dataset.get_patches(FLAGS.test_name)

    labels_train = dataset.get_labels(FLAGS.train_name)
    
    # quick fix in order to have normalized data beforehand
    patches_train = normalize_data(patches_train, mean, std)
    patches_test = normalize_data(patches_test, mean, std)

    # generate triplets ids
    triplets = generate_triplets(labels_train, FLAGS.num_triplets)

    # get matches for evaluation
    matches_train = dataset.get_matches(FLAGS.train_name)
    matches_test = dataset.get_matches(FLAGS.test_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # the random seed must be set at graph level, otherwise you'll be F**k Up :D
        tf.set_random_seed(FLAGS.seed)

        with tf.name_scope('inputs'):
            # Define the input tensor shape
            tensor_shape = (FLAGS.batch_size, FLAGS.image_size,
                            FLAGS.image_size, FLAGS.num_channels)

            # Triplet place holders
            input_anchors = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='anchors')
            input_positives = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='positives')
            input_negatives = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='negatives')

        with tf.name_scope('accuracy'):
            accuracy_pl = tf.placeholder(tf.float32)

        # Creating the architecture
        tfeat_architecture = TFeat(FLAGS.weight_decay, FLAGS.activation)
        tfeat_anchor = tfeat_architecture.model(input_anchors)
        tfeat_positive = tfeat_architecture.model(input_positives)
        tfeat_negative = tfeat_architecture.model(input_negatives)
        
        # Add to the Graph the Ops for loss calculation.
        loss_op, positives, negatives = compute_triplet_loss(
            tfeat_anchor, tfeat_positive, tfeat_negative, FLAGS.margin)

        # Defining training parameters
        step = tf.Variable(0, trainable=False)

        # Define the optimizer to use
        if FLAGS.optimizer == 'momentum':
            # Add to the Graph the Ops for optimization
            train_op = tf.train.MomentumOptimizer(
                learning_rate=FLAGS.learning_rate, momentum=0.9) \
                .minimize(loss_op, global_step=step)
        elif FLAGS.optimizer == 'adam':
            train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) \
                .minimize(loss_op, global_step=step)
        else:
            raise Exception("[ERROR] Optimizer '{}' not supported"
                            .format(FLAGS.optimizer))

        # Build the summary operation based on the TF collection of Summaries.
        if tf.__version__ == '0.11.0':
            tf.scalar_summary('loss', loss_op)
            tf.scalar_summary('positives', positives)
            tf.scalar_summary('negatives', negatives)
            summary_op = tf.merge_all_summaries()

            accuracy_op = tf.scalar_summary('accuracy', accuracy_pl)
        else:
            tf.summary.scalar('loss', loss_op)
            tf.summary.scalar('positives', positives)
            tf.summary.scalar('negatives', negatives)
            summary_op = tf.summary.merge_all()

            accuracy_op = tf.summary.scalar('accuracy', accuracy_pl)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create the op for initializing variables.
        init_op = tf.initialize_all_variables()

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        # Run the Op to initialize the variables.
        session.run(init_op)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        if tf.__version__ == '0.11.0':
            summary_writer_train = tf.train.SummaryWriter(
                os.path.join(LOG_DIR, 'train'), session.graph)
            summary_writer_test = tf.train.SummaryWriter(
                os.path.join(LOG_DIR, 'test'), session.graph)
        else:
            summary_writer_train = tf.summary.FileWriter(
                os.path.join(LOG_DIR, 'train'), session.graph)
            summary_writer_test = tf.summary.FileWriter(
                os.path.join(LOG_DIR, 'test'),  session.graph)

        # And then after everything is built, start the training loop.
        global_step = 0
        for epoch in xrange(FLAGS.num_epochs):
            print('#############################')
            print('Epoch: {}'.format(epoch))

            # shuffle training set
            np.random.shuffle(triplets)

            # train
            print('Training ...')
            run_model(session, global_step, patches_train, triplets,
                      FLAGS.batch_size, input_anchors, input_positives,
                      input_negatives, train_op, loss_op, summary_op,
                      summary_writer_train)

            # update global step
            global_step += len(triplets) // FLAGS.batch_size

            # accuracy: train
            print('Accuracy train ...')
            eval_model(session, global_step, patches_train, matches_train,
                       FLAGS.batch_size, input_anchors, input_positives,
                       tfeat_anchor, tfeat_positive, accuracy_pl,
                       accuracy_op, summary_writer_train)

            # accuracy: test
            print('Accuracy test ...')
            eval_model(session, global_step, patches_test, matches_test,
                       FLAGS.batch_size, input_anchors, input_positives,
                       tfeat_anchor, tfeat_positive, accuracy_pl,
                       accuracy_op, summary_writer_test)

            # Save a checkpoint periodically.
            print('Saving')
            file_name = os.path.join(LOG_DIR, 'model_{}'
                                     .format(FLAGS.train_name))
            saver.save(session, file_name, global_step=global_step)

            print('Done training for {} epochs, {} steps.'
                  .format(epoch, global_step))
        
        # Wait for threads to finish.
        session.close()


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
        tf.gfile.MkDir(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)  # fix the directory to be created
    run_training()


if __name__ == '__main__':
    tf.app.run()
