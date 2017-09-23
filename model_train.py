import tensorflow as tf
from market1501_input import make_slim_dataset,input_fn
from preprocessing import preprocess_image
from nets import nets_factory
import time
import os


slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)

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
    'weight_decay', 0.0005, 'The weight decay on the model weights.')


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



def get_restore_variabels():
    exclusions = []

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

    return variables_to_restore


def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))




def main(_):




    with tf.Graph().as_default():
        #############
        # data set
        ############
        record_file = os.path.join(
            FLAGS.dataset_dir,'%s%s.tfrecord'%(FLAGS.dataset_name,FLAGS.dataset_split_name))
        images,labels,_ = input_fn(record_file,is_training=True)
        #images,labels,_ = input_fn()
        #dataset=make_slim_dataset(FLAGS.dataset_split_name, FLAGS.dataset_dir)
        #provider = slim.dataset_data_provider.DatasetDataProvider(dataset,shuffle=True)
        #image,label = provider.get(['image','label'])

        ################
        # select network
        ################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=751,
            weight_decay=FLAGS.weight_decay,
            is_training=True
        )

        labels = slim.one_hot_encoding(labels, 751)
        logits, end_points = network_fn(images)
        if 'AuxLogits' in end_points:
            tf.losses.softmax_cross_entropy(
                    logits=end_points['AuxLogits'], onehot_labels=labels,
                    label_smoothing=0, weights=0.4, scope='aux_loss')
        tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels,
                label_smoothing=0, weights=1.0)
        total_loss = tf.losses.get_total_loss()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.MomentumOptimizer(0.001,0.9)
        train_op = optimizer.minimize(total_loss,global_step=tf.train.get_global_step())


        variabels_to_restore = get_restore_variabels()
        restore_saver = tf.train.Saver(variabels_to_restore,max_to_keep=4)
        saver = tf.train.Saver()
        mean_loss = 0
        with tf.Session() as sess:
            #initialize_uninitialized_vars(sess)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            restore_saver.restore(sess,FLAGS.checkpoint_path)
            start_time = time.time()
            for step in range(FLAGS.max_number_of_steps):
                try:
                    _,loss = sess.run([train_op,total_loss])
                    mean_loss += loss
                    if (step+1) % 20 == 0:
                        left_seconds = (time.time()-start_time)/step * (FLAGS.max_number_of_steps - step)
                        tf.logging.info('step: {}, loss: {}, time left: {}'.format(step,mean_loss/20,time.strftime('%H:%M:%S',time.gmtime(left_seconds))))
                        mean_loss = 0
                    if (step+1) % 2000 == 0:
                        saver.save(sess,'%s/model.ckpt'%FLAGS.train_dir,global_step=step+1)
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    tf.app.run()
