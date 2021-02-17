import os
import subprocess
from glob import glob

from absl import app
from absl import flags

from face_utils import *


FLAGS = flags.FLAGS

flags.DEFINE_string("openface_dir", "../../../OpenFace/", "path to OpenFace dir")
flags.DEFINE_string("img_dir", "../dataset/images/", "path to images-stored dir")
flags.DEFINE_string("cropped_img_dir", "../dataset/cropped/", "path to cropped-image dir")


def check_dir(dpath):
    assert os.path.isdir(dpath), "specified dir doesn't exist. You specified {}" \
                                 .format(dpath)


def move_csvs(to_dir):
    command = "for csv in processed/*.csv; do mv \"$csv\" {}; done".format(to_dir)
    subprocess.run(command, shell=True)


def remove_unneeded_dir():
    command = "rm -r processed"
    subprocess.run(command, shell=True)


def main(argv):
    print("Start preprocessing.")
    
    check_dir(FLAGS.openface_dir)
    check_dir(FLAGS.img_dir)
    os.makedirs(FLAGS.cropped_img_dir, exist_ok=True)
    detect_crop_and_save_faces(FLAGS.img_dir, FLAGS.cropped_img_dir)
    get_aus(FLAGS.openface_dir, FLAGS.img_dir)
    move_csvs(FLAGS.cropped_img_dir)
    remove_unneeded_dir()

    print("Preprocessing is done!")


if __name__ == "__main__":
    app.run(main)
