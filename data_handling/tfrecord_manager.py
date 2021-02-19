import os
import math

from tqdm import tqdm
from absl import app
from absl import flags

import tensorflow as tf

from .data_loader import get_data


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "../dataset/cropped/", "path to the dataset dir")
flags.DEFINE_string("out_dir", "../dataset/tfrecords/", "path for the output tfrecords")


## For TFRecords
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature_list(list_val):
    features = [tf.train.Feature(
        float_list=tf.train.FloatList(value=[val])) for val in list_val]
    return tf.train.FeatureList(feature=features)


def convert_data_to_tfrecord(imgs, labels, num_split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Start converting data into TFRecords.\n")
    num_data = len(imgs)
    num_per_shard = math.ceil(num_data / num_split)

    for shard_id in tqdm(range(num_split)):
        out_fname = os.path.join(out_dir, 
                                 "celeb_a-{:02d}-of-{:02d}.tfrecord".format(shard_id,
                                                                            num_split))
        
        with tf.io.TFRecordWriter(out_fname) as writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_data)
            for i in range(start_idx, end_idx):
                example = image_label_example(imgs[i], labels[i])
                writer.write(example.SerializeToString())

    print("Finished converting data into TFRecords.")


def image_label_example(img_path, label):
    img_string = open(img_path, 'rb').read()
    height, width, channel = tf.image.decode_jpeg(img_string).shape

    feature = {
        "image": _bytes_feature(img_string),
        "height": _int64_feature(height),
        "width": _int64_feature(width),
        "channel": _int64_feature(channel),
    }

    feature_list = {
        "label": _float_feature_list(label),
    }

    return tf.train.SequenceExample(context=tf.train.Features(feature=feature),
                                    feature_lists=tf.train.FeatureLists(
                                                    feature_list=feature_list))


def parse_tfrecords(example_proto):
    # Parse the input tf.train.Example proto using the dictionaries below
    feature_desc = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
    }

    feature_list_desc = {
        "label": tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    }

    inp, trg = tf.io.parse_single_sequence_example(example_proto, 
                                                   context_features=feature_desc,
                                                   sequence_features=feature_list_desc)

    img = tf.io.decode_jpeg(inp["image"])
    label = trg["label"]

    return img, label


def main(argv):
    ipath_list, ini_lbl_list = get_data(FLAGS.data_dir)
    convert_data_to_tfrecord(ipath_list, ini_lbl_list, 10, FLAGS.out_dir)

if __name__ == "__main__":
    app.run(main)