import tensorflow as tf
from market1501_input import make_slim_dataset,input_fn
from preprocessing import preprocessing_factory
from nets import nets_factory
import os


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/checkpoints/market-1501',
    'Directory where checkpoints and event logs are written to.')


tf.app.flags.DEFINE_string(
    'dataset_name', 'market-1501_', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/tmp/Market-1501/', 'The directory where the dataset files are stored.')


tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')


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

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/checkpoints/inception_v3.ckpt',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'InceptionV3/Logits,InceptionV3/AuxLogits',
    'The path to a checkpoint from which to fine-tune.')



######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')


FLAGS = tf.app.flags.FLAGS


def _get_init_fn():


    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
    An init function run by the supervisor.
    """

    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
            for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore)


def main(_):




    with tf.Graph().as_default():
        #############
        # data set
        ############
        record_file = os.path.join(
            FLAGS.dataset_dir,'%s%s.tfrecord'%(FLAGS.dataset_name,FLAGS.dataset_split_name))
        image,label,_ = input_fn(record_file,is_training=True)
        #dataset=make_slim_dataset(FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ################
        # select network
        ################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=751,
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

        train_image_size = network_fn.default_image_size

        image = image_preprocessing_fn(image,train_image_size,train_image_size)

        images, labels = tf.train.batch(
            [image,label],
            batch_size=FLAGS.batch_size,
            capacity=5 * FLAGS.batch_size
        )
        labels = slim.one_hot_encoding(
          labels, 751)

        logits, end_points = network_fn(images)


        if 'AuxLogits' in end_points:
            tf.losses.softmax_cross_entropy(
                logits=end_points['AuxLogits'], onehot_labels=labels,
                label_smoothing=0, weights=0.4, scope='aux_loss')
        tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels,
            label_smoothing=0, weights=1.0)

        total_loss = slim.losses.get_total_loss()

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.01)

        train_op = slim.learning.create_train_op(total_loss,optimizer)



        slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            init_fn=_get_init_fn()
        )




if __name__ == '__main__':
    tf.app.run()
