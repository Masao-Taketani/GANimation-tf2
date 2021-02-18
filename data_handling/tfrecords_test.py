import os
import sys

import tensorflow as tf

from tfrecord_manager import parse_tfrecords
sys.path.append("..")
from utils import preprocess_for_training


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    tfrecords_dir = "../dataset/tfrecords/"
    batch_size = 25

    tfr_dataset = tf.data.Dataset.list_files(os.path.join(tfrecords_dir, "*.tfrecord"))
    tfr_dataset = tfr_dataset.interleave(tf.data.TFRecordDataset,
                                         num_parallel_calls=AUTOTUNE,
                                         deterministic=False)
    tfr_dataset = tfr_dataset.map(parse_tfrecords)
    tfr_dataset = tfr_dataset.map(preprocess_for_training,
                                  num_parallel_calls=AUTOTUNE)
    tfr_dataset = tfr_dataset.batch(batch_size=batch_size)
    tfr_dataset = tfr_dataset.prefetch(buffer_size=AUTOTUNE)

    imgs, ini_conds, fin_conds = next(iter(tfr_dataset.take(1)))
    print("imgs.shape", imgs.shape)
    print("imgs\n", imgs[0])
    print("ini_cond.shape", ini_conds.shape)
    print("ini_cond", ini_conds[0])
    print("fin_cond.shape", fin_conds.shape)
    print("fin_cond", fin_conds[0])