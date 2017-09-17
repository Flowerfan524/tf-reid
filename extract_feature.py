import tensorflow as tf
from market1501_input import input_fn
from nets import nets_factory
from preprocessing import preprocess_image
import os
import numpy as np
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
        'split_name', 'query',
        'The name of query/test split'
        )
tf.app.flags.DEFINE_string(
        'model_name', 'inception_v3',
        'The name of model'
        )
FLAGS = tf.app.flags.FLAGS

feature_map = {
                'vgg_16': 'fc7_end_points',
                'inception_v3':'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0',
                'inception_v4': 'PreLogitsFlatten',
                'inception_resnet_v2': 'PreLogitsFlatten',
                'resnet_v1_50': 'resnet_v1_50/pool5:0',
                'resnet_v2_50': 'pool5_end_points',
                'mobilenet_v1': 'AvgPool_1a',
               }


def extract_features(model_name,record_file,checkpoints):
    features = []
    classes = []
    cameras = []
    with tf.Graph().as_default():
        image,label,cam = input_fn(record_file)
        network_fn = nets_factory.get_network_fn(model_name,num_classes=751)
        train_image_size = network_fn.default_image_size
        image = preprocess_image(image,train_image_size,train_image_size)
        images,labels,cams = tf.train.batch([image,label,cam],batch_size=32)
        #saver = tf.train.import_meta_graph('%s/model.ckpt-38000.meta'%checkpoints)
        logits,_ = network_fn(images)
        if model_name not in feature_map: raise ValueError('model do not exist')
        feature_name = feature_map[model_name]
        with tf.Session() as sess:
            feature = sess.graph.get_tensor_by_name(feature_name)
            feature = tf.squeeze(feature)
            saver = tf.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(checkpoints))

        #init_fn = slim.assign_from_checkpoint_fn(
        #    checkpoints,
        #    slim.get_variables_to_restore())
        
        #sv = tf.train.Supervisor(logdir=checkpoints)
        #with sv.managed_session() as sess:
        #with tf.Session() as sess:
        #    init_op = tf.global_variables_initializer()
        #    sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            while True:
                try:
                    np_feature,np_label,np_cam = sess.run([feature,labels,cams])
                    features += [np_feature]
                    classes += [np_label]
                    cameras += [np_cam]
                except tf.errors.OutOfRangeError:
                    break
            coord.request_stop()
            coord.join(threads)
        features = np.reshape(features,[-1,feature.shape[-1]])
        classes = np.reshape(classes,[-1,1])
        cameras = np.reshape(cameras,[-1,1])
        return features,classes,cameras

def main(_):
    split_name = FLAGS.split_name
    model_name=FLAGS.model_name
    record_file='/tmp/Market-1501/market-1501_%s.tfrecord'%split_name
    checkpoints='/tmp/checkpoints/market-1501/%s'%model_name
    feature,label,cam = extract_features(model_name, record_file, checkpoints)
    np.savez('/tmp/Market-1501/feature/%s'%split_name,
            feature=feature,
            label=label,
            cam=cam)


if __name__ == '__main__':
    tf.app.run()
