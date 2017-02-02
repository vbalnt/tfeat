import tensorflow as tf

from models.tfeat import TFeat
from datasets.ubc import UBCDataset

from scripts.process import eval_model
from scripts.preprocess import normalize_data, load_stats

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

    # Load mean and std
    print('Loading training stats:')
    mean, std = load_stats(FLAGS.train_name)
    print('-- Mean: %s' % mean)
    print('-- Std:  %s' % std)
    
    # get patches
    patches = dataset.get_patches(FLAGS.test_name)
    matches = dataset.get_matches(FLAGS.test_name)

    # quick fix in order to have normalized data beforehand
    patches = normalize_data(patches, mean, std)

    with tf.Graph().as_default():
        with tf.name_scope('inputs'):
            # Define the input tensor shape
            tensor_shape = (FLAGS.batch_size, FLAGS.image_size,
                            FLAGS.image_size, FLAGS.num_channels)

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

    with tf.Session() as sess:
        # restore session from file
        saver = tf.train.import_meta_graph('%s.meta' % FLAGS.model_file)
        saver.restore(sess, FLAGS.model_file)

        # accuracy: test
        print('Accuracy test ...')
        eval_model(sess, 0, patches, matches,
                   FLAGS.batch_size, inputs1_pl, inputs2_pl,
                   tfeat_inputs1, tfeat_inputs2, None,
                   None, None)


def main(_):
    run_evaluation()


if __name__ == '__main__':
    tf.app.run()
