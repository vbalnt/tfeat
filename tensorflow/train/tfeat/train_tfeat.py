import tensorflow as tf
import numpy as np

import os
import time
import pickle
import random

from tqdm import tqdm

from loss import *
from models.tfeat import TFeat
from datasets.ubc import UBCDataset

from scripts.preprocess import normalize_data
from scripts.eval_metrics import ErrorRateAt95Recall

# Define the algorithm flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('image_size', 32,
                            """The size of the images to process""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """The number of channels in the images to process""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """The size of the mini-batch""")
tf.app.flags.DEFINE_integer('num_epochs', 60,
                            """The number of iterations during the training""")
tf.app.flags.DEFINE_float('margin', 1.0,
                          """The margin value for the loss function""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          """The learning rate for the SGD optimization""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/patches_dataset',
                           """The default path to the patches dataset""")
tf.app.flags.DEFINE_string('train_name', 'notredame',
                           """The default dataset name for training""")
tf.app.flags.DEFINE_string('test_name', 'liberty',
                           """The default dataset name for testing""")
tf.app.flags.DEFINE_integer('num_triplets', 1280000,
                            """The default number of pairs to generate""")
tf.app.flags.DEFINE_string('gpu_id', '0',
                           """The default GPU id to use""")
tf.app.flags.DEFINE_integer('seed', 666,
                            """random seed (default: 666)""")

# to use only a single gpu
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id

# set seed for all the random stuff
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

def run_training():
    # load data
    dataset = UBCDataset(FLAGS.data_dir)
    dataset.load_by_name(FLAGS.train_name)
    dataset.load_by_name(FLAGS.test_name)

    # compute mean and std
    #mean, std = dataset.generate_stats(train_name)
    print('Loading training stats:')
    file = open('../../data/stats_%s.pkl' % FLAGS.train_name, 'r')
    mean, std = pickle.load(file)
    print('-- Mean: %s' % mean)
    print('-- Std:  %s' % std)
    
    # get patches
    patches_train = dataset._get_patches(FLAGS.train_name)
    patches_test  = dataset._get_patches(FLAGS.test_name)
    
    # quick fix in order to have normalized data beforehand
    patches_train = normalize_data(patches_train, mean, std)
    patches_test  = normalize_data(patches_test , mean, std)

    # generate triplets ids
    triplets = dataset.generate_triplets(FLAGS.train_name, FLAGS.num_triplets)

    # get matches for evaluation
    matches_train = dataset._get_matches(FLAGS.train_name)
    matches_test  = dataset._get_matches(FLAGS.test_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        with tf.name_scope('inputs'):
            # User defined parameters
            BATCH_SIZE     = FLAGS.batch_size
            NUM_CHANNELS   = FLAGS.num_channels
            IMAGE_SIZE     = FLAGS.image_size

            # Define the input tensor shape
            tensor_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

            # Triplet place holders
            input_anchors   = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='anchors')
            input_positives = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='positives')
            input_negatives = tf.placeholder(
                dtype=tf.float32, shape=tensor_shape, name='negatives')

        with tf.name_scope('accuracy'):
            accuracy_pl = tf.placeholder(tf.float32)

        # Creating the architecture
        tfeat_architecture = TFeat()
        tfeat_anchor   = tfeat_architecture.model(input_anchors)
        tfeat_positive = tfeat_architecture.model(input_positives)
        tfeat_negative = tfeat_architecture.model(input_negatives)
        
        # Add to the Graph the Ops for loss calculation.
        loss, positives, negatives = compute_triplet_loss(
            tfeat_anchor, tfeat_positive, tfeat_negative, FLAGS.margin)

        # Defining training parameters
        step = tf.Variable(0, trainable=False)        
                
        '''learning_rate = tf.train.exponential_decay(
					FLAGS.learning_rate, # Base learning rate.
					global_step=step,
					decay_steps= FLAGS.batch_size,
					decay_rate=1-0.000001,
					staircase=False)
        
        # Add to the Graph the Ops for optmization
        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9) \
                .minimize(loss, global_step=step)'''

        # Add to the Graph the Ops for optmization
        train_op = tf.train.MomentumOptimizer(
            learning_rate=FLAGS.learning_rate, momentum=0.9) \
                .minimize(loss, global_step=step)
        #train_op = tf.train.AdamOptimizer(1e-4).minimize(loss, global_step=step)


        # Build the summary operation based on the TF collection of Summaries.
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('positives', positives)
        tf.summary.scalar('negatives', negatives)
        summary_op = tf.summary.merge_all()

        accuracy_op = tf.summary.scalar('accuracy', accuracy_pl)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create the op for initializing variables.
        init_op = tf.initialize_all_variables()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Run the Op to initialize the variables.
        sess.run(init_op)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train = tf.summary.FileWriter(
            './logs_tensorboard/triplet_relu/train', sess.graph)
        summary_writer_test  = tf.summary.FileWriter(
            './logs_tensorboard/triplet_relu/test',  sess.graph)

        def run_model(sess, global_step, data, data_ids, batch_size, train=False):
            pbar = tqdm(xrange(len(data_ids) // batch_size))
            for step in pbar:
                start_time = time.time()

                # get data batch
                batch_anchors, batch_positives, batch_negatives = \
                    dataset.get_batch(data, data_ids, step, batch_size)
                    
                '''import cv2
                for x in xrange(0, BATCH_SIZE):
                norm = preprocess_data(batch_anchors[x],mean,std)
                print np.mean(norm)
                pt = np.hstack([batch_anchors[x], batch_positives[x], batch_negatives[x]])
                cv2.imshow('patches', pt)
                cv2.waitKey(0)'''
                
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.
                feed_dict = {
                    input_anchors:   batch_anchors,
                    input_positives: batch_positives,
                    input_negatives: batch_negatives
                }

                # Run one step of the model.
                if train:
                    _, loss_value, summary = sess.run(
                        [train_op, loss, summary_op], feed_dict=feed_dict)

                    # Update the events file.
                    summary_writer_train.add_summary(summary, global_step)
                    # update step
                    global_step += 1
                else:
                    loss_value, summary = sess.run(
                        [loss, summary_op], feed_dict=feed_dict)
                    # Update the events file.
                    summary_writer_test.add_summary(summary, global_step)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                # Print status to stdout.
                if global_step % 100 == 0:
                    pbar.set_description('loss = %.6f (%.3f sec)'
                        % (loss_value, duration))

        def eval_model(sess, data, matches, global_step, summary_writer):
            offset = 0
            dists  = np.zeros(matches.shape[0],)
            labels = np.zeros(matches.shape[0],)

            for x in tqdm(xrange(matches.shape[0] // FLAGS.batch_size)):
                # get data batch
                batch = matches[offset:offset + FLAGS.batch_size, :]

                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.
                feed_dict = {
                    input_anchors:   data[batch[:,0]],
                    input_positives: data[batch[:,1]]
                }

                # Run one step of the model. # Run the graph and fetch some of the nodes.
                descs1, descs2 = sess.run([tfeat_anchor, tfeat_positive],
                    feed_dict=feed_dict) 

                # compute euclidean distances between descriptors
                for i in xrange(FLAGS.batch_size):
                    idx = x * FLAGS.batch_size + i
                    dists[idx]  = np.linalg.norm(descs1[i,:] - descs2[i,:])
                    labels[idx] = batch[i,2]

            # compute the false positives rate
            fpr95 = ErrorRateAt95Recall(labels, dists)
            print 'FRP95: %s' % fpr95

            acc_val = sess.run(accuracy_op, feed_dict={ accuracy_pl: fpr95 })
            summary_writer.add_summary(acc_val, global_step)


        # And then after everything is built, start the training loop.
        global_step = 0
        for epoch in xrange(FLAGS.num_epochs):
            print('#############################')
            print('Epoch: %s' % epoch)

            epoch_time = time.time()

            # shuffle training set
            random.shuffle(triplets)
           
            # train
            print('Training ...')
            run_model(sess, global_step, patches_train,
                triplets, FLAGS.batch_size, True)
            
            # update global step
            global_step += len(triplets) // FLAGS.batch_size
            
            # accuray: train
            print('Accuracy train ...')
            eval_model(sess, patches_train, matches_train, \
                       global_step, summary_writer_train)

            # accuray: test
            print('Accuracy test ...')
            eval_model(sess, patches_test, matches_test, \
                       global_step, summary_writer_test)

            # Save a checkpoint periodically.
            print('Saving')
            fname = 'model_%s' % (FLAGS.train_name)
            saver.save(sess, fname, global_step=global_step)

            print('Done training for %d epochs, %d steps.' % (epoch, global_step))
        
        # Wait for threads to finish.
        sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
