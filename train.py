
import os
import time
import datetime

from tqdm import tqdm
from absl import app
from absl import flags

import tensorflow as tf

from models import build_model


FLAGS = flags.FLAGS

flags.DEFINE_string("ckpt_dir", "ckpts/", "path to the checkpoint dir")
flags.DEFINE_string("logdir", "logs/", "path to the log dir")
flags.DEFINE_string("test_result_dir", "test_results/", "path to the test result dir")
flags.DEFINE_integer("num_epochs", 30, "number of epopchs to train")
flags.DEFINE_integer("num_epochs_decay", 20, "number of epochs to start lr decay")
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

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    os.makedirs(FLAGS.ckpt_dir, exist_ok=True)
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs(FLAGS.test_result_dir, exist_ok=True)

    gen, disc = build_model(FLAGS.num_cond)

    gen_opt = tf.keras.optimizers.Adam(FLAGS.g_lr, FLAGS.beta1, FLAGS.beta2)
    disc_opt = tf.keras.optimizers.Adam(FLAGS.d_lr, FLAGS.beta1, FLAGS.beta2)

    # Set the checkpoint and the checkpoint manager.
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0, dtype=tf.int64),
                               gen=gen,
                               disc=disc,
                               gen_opt=gen_opt,
                               disc_opt=disc_opt)

    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              FLAGS.ckpt_dir,
                                              max_to_keep=5)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint is restored!")

    # Create a summary writer to track the losses
    summary_writer = tf.summary.create_file_writer(
                                    os.path.join(FLAGS.logdir,
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                                )

    d_loss_list, g_loss_list = initialize_loss_trackers()

    # Train the discriminator and the generator
    while ckpt.epoch < FLAGS.num_epochs:
        ckpt.epoch.assign_add(1)
        step = tf.constant(0)
        reset_loss_trackers(d_loss_list)
        reset_loss_trackers(g_loss_list)

        start = time.time()
        for x_real, label_org, label_trg in tqdm(train_dataset):
            step += 1
            #if step.numpy() > FLAGS.num_iters_decay:
            #    update_lr_by_iter(gen_opt, disc_opt, step, diff_iter, FLAGS.g_lr, FLAGS.d_lr)

            d_losses = train_disc(disc,
                                  gen,
                                  x_real,
                                  label_ini_cond, 
                                  label_fin_cond, 
                                  lambda_cond, 
                                  lambda_gp, 
                                  disc_opt)

            update_loss_trackers(d_loss_list, d_losses)

            if step.numpy() % FLAGS.num_critic_updates == 0:
                g_losses = train_gen(disc,
                                     gen,
                                     x_real,
                                     label_ini_cond.
                                     label_fin_cond,
                                     lambda_cond,
                                     lambda_rec,
                                     lambda_attn,
                                     lambda_tv,
                                     gen_opt)

                update_loss_trackers(g_loss_list, g_losses)

            if step.numpy() == iters_per_epoch:
                break

        end = time.time()
        print_log(ckpt.epoch.numpy(), start, end, d_losses, g_losses)

        # keep the log for the losses
        with summary_writer.as_default():
            tf.summary.scalar("d_loss", d_loss_list[0].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_cond", d_loss_list[1].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_real", d_loss_list[2].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_fake", d_loss_list[3].result(), step=ckpt.epoch)
            tf.summary.scalar("d_loss_gp", d_loss_list[4].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss", g_loss_list[0].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_fake", g_loss_list[1].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_cond", g_loss_list[2].result(), step=ckpt.epoch)
            tf.summary.scalar("g_loss_rec", g_loss_list[3].result(), step=ckpt.epoch)
            tf.summary.scalar("g_fake_attn_mask_loss", g_loss_list[0].result(), step=ckpt.epoch)
            tf.summary.scalar("g_rec_attn_mask_loss", g_loss_list[1].result(), step=ckpt.epoch)
            tf.summary.scalar("g_fake_tv_loss", g_loss_list[2].result(), step=ckpt.epoch)
            tf.summary.scalar("g_rec_tv_loss", g_loss_list[3].result(), step=ckpt.epoch)

        # test the generator model and save the results for each epoch
        fpath = os.path.join(FLAGS.test_result_dir, "{}-images.jpg".format(ckpt.epoch.numpy()))
        save_test_results(gen, test_imgs[:FLAGS.num_test], c_fixed_trg_list, fpath)

        if (ckpt.epoch) % FLAGS.model_save_epoch == 0:
            ckpt_save_path = ckpt_manager.save()
            print("Saving a checkpoint for epoch {} at {}".format(ckpt.epoch.numpy(), ckpt_save_path))

        if ckpt.epoch > FLAGS.num_epochs_decay:
            update_lr(gen_opt, disc_opt, FLAGS.num_epochs, ckpt.epoch, FLAGS.g_lr, FLAGS.d_lr)



def __name__ == "__main__":
    app.run(main)