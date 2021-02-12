
import os
import time
import datetime

from tqdm import tqdm
from absl import app
from absl import flags

import tensorflow as tf

from models import build_model


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 30, "number of epopchs to train")
flags.DEFINE_integer("num_epochs_decay", 10, "number of epochs to start lr decay")
flags.DEFINE_integer("model_save_epoch", 1, "to save model every specified epochs")
flags.DEFINE_integer("num_critic_updates", 5, "number of a Discriminator updates "
                                              "every time a generator updates")
flags.DEFINE_integer("num_cond", 17, "number of conditions")
flags.DEFINE_float("g_lr", 0.0001, "learning rate for the generator")
flags.DEFINE_float("d_lr", 0.0001, "learning rate for the discriminator")
flags.DEFINE_float("beta1", 0.5, "beta1 for Adam optimizer")
flags.DEFINE_float("beta2", 0.999, "beta2 for Adam optimizer")
flags.DEFINE_float("lambda_rec", 10.0, "weight for reconstruction loss")
flags.DEFINE_float("lambda_gp", 10.0, "weight for gradient penalty loss")
flags.DEFINE_float("lambda_attn", 0.1, "weight for attention loss")
flags.DEFINE_float("lambda_cond", 4000, "weight for conditional expression loss")
flags.DEFINE_float("lambda_tv", 1e-5, "weight for total variation loss")


def main(argv):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gen, disc = build_model(FLAGS.num_cond)




def __name__ == "__main__":
    app.run(main)