import tensorflow as tf
from market1501_input import input_fn

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
    with tf.Graph().as_default():
        image,label = input_fn(record_file)
        network_fn = nets_factory.get_network_fn(model_name)
        logits,endpoints = network_fn(image)
        if model_name not in feature_map: raise ValueError('model do not exist')
        feature_name = feature_map[model_name]
        feature = endpoints[feature_name]
        feature = tf.squeeze(feature)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, checkpoints),
            slim.get_variables_to_restore())
        with tf.Session() as sess:
            init_fn(sess)
            while True:
                try:
                    np_feature = sess.run(feature)
                    features += [np_feature]
                except tf.errors.OutOfRangeError:
                    break
        features = np.reshape(features,[-1,feature.shape[-1]])
        return features

def main(_):
    model_name='inception_v3'
    record_file='/tmp/Market-1501/market-1501_query.tfrecord'
    checkpoints='/tmp/Market-1501'
    features = extract_features(model_name, record_file, checkpoints)


if __name__ == '__main__':
    tf.app.run()
