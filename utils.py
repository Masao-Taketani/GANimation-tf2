import tensorflow as tf

from models.losses import *


@tf.function
def train_disc(disc,
               gen,
               x_real,
               label_ini_cond, 
               label_fin_cond, 
               lambda_cond, 
               lambda_gp, 
               opt):

    with tf.GradientTape() as tape:
        # Compute loss with real images
        real_out_vals, real_out_cond = disc(x_real, training=True)
        d_loss_real = - get_mean_for_loss(real_out_vals)
        d_loss_cond = get_conditional_expression_loss(label_ini_cond, real_out_cond)
        # Compute loss with fake images
        x_fake, fake_attn_mask = gen(x_real, label_fin_cond, training=False)
        x_fake_masked = fake_attn_mask * x_real + (1 - fake_attn_mask) * x_fake
        fake_out_vals, _ = disc(x_fake_masked, training=True)
        d_loss_fake = get_mean_for_loss(fake_out_vals)
        # Compute loss for gradient penalty
        d_loss_gp = get_gradient_penalty(x_real, x_fake_masked, disc)
        # Compute the total loss for the discriminator
        d_loss = lambda_cond * d_loss_cond + d_loss_real + d_loss_fake + lambda_gp * d_loss_gp

    # Calculate the gradients and update params for the discriminator and the generator
    disc_gradients = tape.gradient(d_loss, disc.trainable_variables)
    opt.apply_gradients(zip(disc_gradients, disc.trainable_variables))

    return d_loss, d_loss_cond, d_loss_real, d_loss_fake, d_loss_gp


@tf.function
def train_gen(disc,
              gen,
              x_real,
              label_ini_cond.
              label_fin_cond,
              lambda_cond,
              lambda_rec,
              lambda_attn,
              lambda_tv,
              opt):

    with tf.GradientTape() as tape:
        # Compute loss for ini-to-fin
        x_fake, fake_attn_mask = gen(x_real, label_fin_cond, training=True)
        x_fake_masked = fake_attn_mask * x_real + (1 - fake_attn_mask) * x_fake
        fake_gen_out_vals, fake_gen_out_cond = disc(x_fake_masked, training=False)
        g_loss_fake = - get_mean_for_loss(fake_gen_out_vals)
        g_loss_cond = get_conditional_expression_loss(label_fin_cond, fake_gen_out_cond)
        # Compute loss for fin-to-ini
        x_gen_rec, rec_attn_mask = gen(x_fake_masked, label_ini_cond, training=True)
        x_rec_masked = rec_attn_mask * x_fake_masked + (1 - rec_attn_mask) * x_gen_rec
        # Compute the cycle and attention loss
        g_loss_rec = get_l1_loss(x_real, x_rec_masked)
        g_fake_attn_mask_loss, g_rec_attn_mask_loss, g_fake_tv_loss, g_rec_tv_loss = \
            get_attention_loss(fake_attn_mask, rec_attn_mask)
        # Compute the total loss for the generator
        g_loss = g_loss_fake + lambda_cond * g_loss_cond + lambda_rec * g_loss_rec + \
                 lambda_attn * g_fake_attn_mask_loss + lambda_attn * g_rec_attn_mask_loss + \
                 lambda_tv * g_fake_tv_loss + lambda_tv * g_rec_tv_loss

    gen_gradients = tape.gradient(g_loss, gen.trainable_variables)
    opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    return g_loss, g_loss_fake, g_loss_cond, g_loss_rec, g_fake_attn_mask_loss, g_rec_attn_mask_loss, g_fake_tv_loss, g_rec_tv_loss


@tf.function
def update_lr(gen_opt, disc_opt, max_epoch, epoch, g_lr=0.0001, d_lr=0.0001):
    if g_lr != d_lr:
        decayed_lr = get_lr_decay_factor(epoch, max_epoch, g_lr)
        gen_opt.lr.assign(decayed_lr)
        disc_opt.lr.assign(decayed_lr)
        # to debug
        tf.print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))
    else:
        g_decayed_lr =  get_lr_decay_factor(epoch, 
                                            max_epoch,
                                            g_lr)
        d_decayed_lr =  get_lr_decay_factor(epoch, 
                                            max_epoch,
                                            d_lr)
        gen_opt.lr.assign(g_decayed_lr)
        disc_opt.lr.assign(d_decayed_lr)
        # to debug
        print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))


@tf.function
def update_lr_by_iter(gen_opt, disc_opt, iteration, diff_iter, g_lr=0.0001, d_lr=0.0001):
    if g_lr != d_lr:
        decayed_lr = get_lr_decay_factor_by_iter(iteration, diff_iter, g_lr)
        gen_opt.lr.assign(decayed_lr)
        disc_opt.lr.assign(decayed_lr)
        # to debug
        #tf.print("decayed lr G: {}, D: {}".format(gen_opt.lr, disc_opt.lr))
    else:
        g_decayed_lr =  get_lr_decay_factor_by_iter(iteration, 
                                                    diff_iter,
                                                    g_lr)
        d_decayed_lr =  get_lr_decay_factor_by_iter(iteration, 
                                                    diff_iter,
                                                    d_lr)
        gen_opt.lr.assign(g_decayed_lr)
        disc_opt.lr.assign(d_decayed_lr)


def print_log(epoch, start, end, d_losses, g_losses):
    tf.print("\nTime taken for epoch {} is {:.3f} sec\n".format(epoch,
                                                                round(end - start)))
    d_log = "d_loss: {:.3f} (d_loss_cond: {:.3f}, d_loss_real: {:.3f}, "
            "d_loss_fake: {:.3f}, d_loss_gp: {:.3f})"
    g_log = "g_loss: {:.3f} (g_loss_fake: {:.3f}, g_loss_cond: {:.3f}, "
            "g_loss_rec: {:.3f}, g_fake_attn_mask_loss: {:.3f}, "
            "g_rec_attn_mask_loss: {:.3f}, g_fake_tv_loss: {:.3f}, "
            "g_rec_tv_loss: {:.3f})"
    tf.print(d_log.format(d_losses[0],
                          d_losses[1], 
                          d_losses[2], 
                          d_losses[3], 
                          d_losses[4]))
    tf.print(g_log.format(g_losses[0], 
                          g_losses[1], 
                          g_losses[2], 
                          g_losses[3], 
                          g_losses[4], 
                          g_losses[5], 
                          g_losses[6], 
                          g_losses[7]))