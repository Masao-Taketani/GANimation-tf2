import os
import subprocess
from glob import glob

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_string("openface_dir", "../../../OpenFace/", "path to OpenFace dir")
flags.DEFINE_string("img_dir", "../dataset/", "path to images-stored dir")


def check_dir(dpath):
    assert os.path.isdir(dpath), "specified dir doesn't exist. You specified {}" \
                                 .format(dpath)

def get_aus(openface_dir, img_dir):
    exe = os.path.join(openface_dir, "build/bin/FaceLandmarkImg")
    command = [exe, "-fdir", img_dir, "-aus"]
    subprocess.run(command)

def move_csvs(img_dir):
    command = "mv processed/*.csv {}".format(img_dir)
    subprocess.run(command, shell=True)

def remove_unneeded_dir():
    command = "rm -r processed"
    subprocess.run(command, shell=True)

def main(argv):
    print("Start preprocessing.")
    
    check_dir(FLAGS.openface_dir)
    check_dir(FLAGS.img_dir)
    get_aus(FLAGS.openface_dir, FLAGS.img_dir)
    move_csvs(FLAGS.img_dir)
    remove_unneeded_dir()

    print("Preprocessing is done!")


if __name__ == "__main__":
    app.run(main)
