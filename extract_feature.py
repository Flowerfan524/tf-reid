import tensorflow as tf
from market1501_input import *
from nets import nets_factory
from preprocessing import preprocess_image
import os
import numpy as np
import feature_util
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
        'split_name', 'query',
        'The name of query/test split'
        )
tf.app.flags.DEFINE_string(
        'model_name', 'inception_v3',
        'The name of model'
        )
tf.app.flags.DEFINE_string(
        'check_step', '18000',
        'The name of model'
        )
FLAGS = tf.app.flags.FLAGS


def extract_features(model_name,data_dir,checkpoints):
    features = []
    classes = []
    cameras = []
    with tf.Graph().as_default():
        images,labels,cams = img_input_fn(data_dir)
        #images,labels,cams = input_fn(record_file)
        network_fn = nets_factory.get_network_fn(model_name,num_classes=751)
        #train_image_size = network_fn.default_image_size
        #saver = tf.train.import_meta_graph('%s/model.ckpt-38000.meta'%checkpoints)
        logits,_ = network_fn(images)
        feature_name = feature_util.get_last_feature_name(model_name=model_name)
        feature = tf.get_default_graph().get_tensor_by_name(feature_name)
        feature = tf.squeeze(feature)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #saver.restore(sess,tf.train.latest_checkpoint(checkpoints))
            saver.restore(sess,checkpoints)
            while True:
                try:
                    np_feature,np_label,np_cam = sess.run([feature,labels,cams])
                    assert np_feature.shape[0] == np_label.shape[0]
                    features += [np_feature]
                    classes += [np_label]
                    cameras += [np_cam]
                except tf.errors.OutOfRangeError:
                    break
        features = np.concatenate(features)
        classes = np.concatenate(classes)
        cameras = np.concatenate(cameras)
        return features,classes,cameras

def main(_):
    split_name = FLAGS.split_name
    model_name=FLAGS.model_name
    check_step = FLAGS.check_step
    record_file='/tmp/Market-1501/%s'%split_name
    checkpoints='/tmp/checkpoints/market-1501/%s/model.ckpt-%s'%(model_name,check_step)
    feature,label,cam = extract_features(model_name, record_file, checkpoints)
    np.savez('/tmp/Market-1501/feature/%s'%split_name,
            feature=feature,
            label=label,
            cam=cam)


if __name__ == '__main__':
    tf.app.run()
