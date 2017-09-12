import os
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf



def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
# url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
# url = "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz"
# url = "http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz"
checkpoints_dir = '/tmp/checkpoints/'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

download_and_uncompress_tarball(url, checkpoints_dir)
