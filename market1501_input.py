import tensorflow as tf
from PIL import Image
import os
slim = tf.contrib.slim


SPLITS_TO_SIZES = {'train': 12936, 'query':3800, 'test': 10000}

_NUM_CLASSES = 751

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [128 x 64 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _conver_to_records(data_dir,tfrecord_writer):

    label_dict = {}
    value = 0
    filenames = os.listdir(data_dir)
    for filename in filenames:
        if not filename.endswith('.jpg'): continue
        img = Image.open(os.path.join(data_dir,filename))
        img_raw = img.tobytes()
        height = img.height
        width = img.width
        if filename.startswith('-'):
            label = -1
            cam = int(filename[4])
        else:
            label = int(x=filename[:4])
            if label not in label_dict:
                label_dict[label] = value
                value += 1
            label = label_dict[label]
            cam = int(x=filename[6])


        example = tf.train.Example(features=tf.train.Features(feature={
          'img_raw': bytes_feature(img_raw),
          'label': int64_feature(label),
          'height': int64_feature(height),
          'width': int64_feature(width),
          'cam': int64_feature(cam)}))

        tfrecord_writer.write(example.SerializeToString())


def make_slim_dataset(split_name,data_dir):
    reader = tf.TFRecordReader
    keys_to_features = {
      'img_raw': tf.FixedLenFeature((), tf.string, default_value=''),
      'img_fmt': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(image_key='img_raw',format_key='img_fmt',shape=[128, 64, 3]),
    'label': slim.tfexample_decoder.Tensor('label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)

    file_pattern = os.path.join(data_dir,
                'market-1501_{}.tfrecord'.format(split_name))

    return slim.dataset.Dataset(
    data_sources=file_pattern,
    reader=reader,
    decoder=decoder,
    num_samples=SPLITS_TO_SIZES[split_name],
    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
    num_classes=_NUM_CLASSES)

def input_fn(filename,is_training=False):
    dataset = tf.contrib.data.TFRecordDataset([filename])

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "img_raw": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
            "cam": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64))
            }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.decode_raw(parsed["img_raw"],tf.uint8)
        image = tf.reshape(image, [128, 64, 3])
        label = tf.cast(parsed["label"], tf.int32)
        image = tf.cast(image,tf.float32)
        cam = tf.cast(parsed["cam"], tf.int32)

        return image, label,cam

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    #dataset = dataset.batch(32)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    imgs, labels, cams = iterator.get_next()
    return imgs, labels, cams



def main(_):
    train_dir = '/tmp/Market-1501/train/'
    query_dir = '/tmp/Market-1501/query/'
    test_dir = '/tmp/Market-1501/test/'

    training_filename = '{}/market-1501_{}.tfrecord'.format('/tmp/Market-1501','train')
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        _conver_to_records(train_dir, tfrecord_writer)

    quering_filename = '{}/market-1501_{}.tfrecord'.format('/tmp/Market-1501','query')
    with tf.python_io.TFRecordWriter(quering_filename) as tfrecord_writer:
        _conver_to_records(query_dir, tfrecord_writer)

    testing_filename = '{}/market-1501_{}.tfrecord'.format('/tmp/Market-1501/','test')
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        _conver_to_records(test_dir, tfrecord_writer)


if __name__ == '__main__':
    tf.app.run()
