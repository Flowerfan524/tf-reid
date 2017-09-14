import tensorflow as tf
from market1501_input import input_fn
from nets import nets_factory
from preprocessing import preprocessing_factory
import os
import numpy as np
slim = tf.contrib.slim

feature_map = {
                'vgg_16': 'fc7_end_points',
                'inception_v3':'PreLogits',
                'inception_v4': 'PreLogitsFlatten',
                'inception_resnet_v2': 'PreLogitsFlatten',
                'resnet_v1_50': 'pool5_end_points',
                'resnet_v2_50': 'pool5_end_points',
                'mobilenet_v1': 'AvgPool_1a',
               }


def extract_features(model_name,record_file,checkpoints):
    features = []
    classes = []
    cameras = []
    with tf.Graph().as_default():
        image,label,cam = input_fn(record_file)
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(model_name)
        network_fn = nets_factory.get_network_fn(model_name,num_classes=751)
        train_image_size = network_fn.default_image_size
        image = image_preprocessing_fn(image,train_image_size,train_image_size)
        images,labels,cams = tf.train.batch([image,label,cam],batch_size=32)
        logits,endpoints = network_fn(images)
        if model_name not in feature_map: raise ValueError('model do not exist')
        feature_name = feature_map[model_name]
        feature = endpoints[feature_name]
        feature = tf.squeeze(feature)

        #init_fn = slim.assign_from_checkpoint_fn(
        #    checkpoints,
        #    slim.get_variables_to_restore())
        sv = tf.train.Supervisor(logdir=checkpoints)
        with sv.managed_session() as sess:
        #with tf.Session() as sess:
        #    init_op = tf.global_variables_initializer()
        #    sess.run(init_op)
        #    coord = tf.train.Coordinator()
        #    threads = tf.train.start_queue_runners(coord=coord)
            while True:
                try:
                    np_feature,np_label,np_cam = sess.run([feature,labels,cams])
                    features += [np_feature]
                    classes += [np_label]
                    cameras += [np_cam]
                except tf.errors.OutOfRangeError:
                    break
        #    coord.request_stop()
        #    coord.join(threads)
        features = np.reshape(features,[-1,feature.shape[-1]])
        classes = np.reshape(cameras,[-1,1])
        cameras = np.reshape(cameras,[-1,1])
        return features,classes,cameras

def main(_):
    model_name='inception_v3'
    record_file='/tmp/Market-1501/market-1501_test.tfrecord'
    checkpoints='/tmp/Market-1501'
    feature,label,cam = extract_features(model_name, record_file, checkpoints)
    np.savez('/tmp/Market-1501/feature/test',
            feature=feature,
            label=label,
            cam=cam)


if __name__ == '__main__':
    tf.app.run()
