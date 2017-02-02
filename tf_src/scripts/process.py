import time
import numpy as np

from tqdm import tqdm

from sampling import generate_batch
from eval_metrics import ErrorRateAt95Recall


def fill_feed_dict(data_set, data_ids,
                   anchors_pl, positives_pl, negatives_pl,
                   step, batch_size, train):
    # get data batch
    batch_out1, batch_out2, batch_out3 = \
        generate_batch(data_set, data_ids, step, batch_size, train)

    '''import cv2
    for x in xrange(0, BATCH_SIZE):
    norm = preprocess_data(batch_anchors[x],mean,std)
    print np.mean(norm)
    pt = np.hstack([batch_anchors[x], batch_positives[x], batch_negatives[x]])
    cv2.imshow('patches', pt)
    cv2.waitKey(0)'''

    # in case is not training, we feed only two anchors
    # and return the matching labels
    if not train:
        feed_dict = {
            anchors_pl: batch_out1,
            positives_pl: batch_out2,
        }
        return feed_dict, batch_out3

    # This dictionary maps the batch data (as a numpy array) to the
    # node in the graph it should be fed to.
    feed_dict = {
        anchors_pl: batch_out1,
        positives_pl: batch_out2,
        negatives_pl: batch_out3
    }
    return feed_dict


def run_model(sess, global_step, data_set, data_ids, batch_size,
              anchors_pl, positives_pl, negatives_pl,
              train_op, loss_op, summary_op, summary_writer):
    progress_bar = tqdm(xrange(len(data_ids) // batch_size))
    for step in progress_bar:
        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(data_set, data_ids,
                                   anchors_pl, positives_pl, negatives_pl,
                                   step, batch_size, train=True)

        # Run one step of the model. The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss_op], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        # Print status to stdout.
        if global_step % 100 == 0:
            progress_bar.set_description(
                'loss = {:.6f} ({:.3f} sec)'.format(loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

        # update the global step for monitoring
        global_step += 1


def eval_model(sess, global_step, data_set, matches, batch_size,
               anchors_pl, positives_pl, anchors_op, positives_op,
               accuracy_pl, accuracy_op, summary_writer):
    dists = np.zeros(matches.shape[0],)
    labels = np.zeros(matches.shape[0],)

    for step in tqdm(xrange(matches.shape[0] // batch_size)):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict, batch_label = fill_feed_dict(
            data_set, matches, anchors_pl, positives_pl, None, step,
            batch_size, train=False)

        # Run one step of the model. # Run the graph and fetch some of the nodes.
        descs1, descs2 = sess.run([anchors_op, positives_op],
                                  feed_dict=feed_dict)

        # compute euclidean distances between descriptors
        for i in xrange(batch_size):
            idx = step * batch_size + i
            dists[idx] = np.linalg.norm(descs1[i, :] - descs2[i, :])
            labels[idx] = batch_label[i]

    # compute the false positives rate
    fpr95 = ErrorRateAt95Recall(labels, dists)
    print 'FRP95: {}'.format(fpr95)

    if summary_writer is not None:
        acc_val = sess.run(accuracy_op, feed_dict={accuracy_pl: fpr95})
        summary_writer.add_summary(acc_val, global_step)
