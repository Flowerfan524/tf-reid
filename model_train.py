import tensorflow as tf
from market1501_input import make_slim_dataset
from preprocessing import preprocessing_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_string(
    'dataset_name', 'market-1501_', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/tmp/Market-1501/', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', 12936, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/checkpoints/inception_v3.ckpt',
    'The path to a checkpoint from which to fine-tune.')


FLAGS = tf.app.flags.FLAGS

def main():


    dataset=make_slim_dataset(FLAGS.dataset_split_name, FLAGS.dataset_dir)


    with tf.Graph().as_default():
        # data set

        ################
        # select network
        ################
        network_fn = slim.nets.nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True
        )

        ###############################
        # select preprocessing function
        ###############################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True
        )

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size
        )
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset

        train_image_size = network_fn.default_image_size

        image = image_preprocessing_fn(image,train_image_size,train_image_size)

        images, labels = tf.train.batch(
            [image,label],
            batch_size=FLAGS.batch_size,
            capacity=5 * FLAGS.batch_size
        )
        labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)

        logits, end_points = network_fn(images)

        slim.losses.softmax_cross_entropy(logits, labels)

        total_loss = slim.losses.get_total_loss()

        optimizer = tf.train.GradientDescentOptimizer(learning_ratae)

        train_op = slim.learning.create_train_op(total_loss,optimizer)

        variables_to_restore = slim.get_model_variables()

        init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint_path, variables_to_restore)


        slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps
            init_fn=init_fn
        )

        with slim.arg_scope():
            logits, endpoints = slim.nets.inception()



if __name__ == '__main__':
    tf.app.run()
