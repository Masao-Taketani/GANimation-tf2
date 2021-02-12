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

    return d_loss_real, d_loss_fake, d_loss_gp, d_loss_cls, d_loss


@tf.function
def train_gen(disc,
              gen,
              x_real,
              label_fin_cond, 
              lambda_cls,
              lambda_rec,
              opt):

    with tf.GradientTape() as tape:
        # Compute loss for original-to-target domain
        x_fake, fake_mask = gen(x_real, label_fin_cond, training=True)
        x_fake_masked = fake_mask * x_real + (1 - fake_mask) * x_fake
        fake_gen_out_vals, fake_gen_out_cond = disc(x_fake_masked, training=False)
        g_loss_fake = - get_mean_for_loss(gen_out_src)
        g_loss_cls = get_classification_loss(label_trg, gen_out_cls)
        # Compute loss for target-to-original domain
        x_rec = gen(x_fake, label_trg, training=True)
        g_loss_rec = get_l1_loss(x_real, x_rec)
        # Compute the total loss for the generator
        g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls

    gen_gradients = tape.gradient(g_loss, gen.trainable_variables)
    opt.apply_gradients(zip(gen_gradients, gen.trainable_variables))

    return g_loss_fake, g_loss_rec, g_loss_cls, g_loss