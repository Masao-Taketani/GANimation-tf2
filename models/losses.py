import tensorflow as tf


def initialize_loss_trackers():
    d_losses = ("d_loss",
                "d_loss_cond",
                "d_loss_real",
                "d_loss_fake",
                "d_loss_gp")

    g_losses = ("g_loss",
                "g_loss_fake",
                "g_loss_cond",
                "g_loss_rec",
                "g_fake_attn_mask_loss",
                "g_rec_attn_mask_loss",
                "g_fake_tv_loss",
                "g_rec_tv_loss")

    d_loss_list = []
    g_loss_list = []
    dl_list = store_loss_tracker(d_loss_list, d_losses)
    gl_list = store_loss_tracker(g_loss_list, g_losses)

    return dl_list, gl_list


def store_loss_tracker(loss_list, losses):
    for name in losses:
        loss_list.append(define_loss_tracker(name))

    return loss_list


def define_loss_tracker(name):
    return tf.keras.metrics.Mean(name=name)


def reset_loss_trackers(loss_list):
    for loss in loss_list:
        loss.reset_states()

    
def update_loss_trackers(loss_tracker_list, losses):
    for tracker, loss in zip(loss_tracker_list, losses):
        tracker(loss)


def get_gradient_penalty(x, x_gen, discriminator):
    # shape=[x.shape[0], 1, 1, 1] to generate a random number for every sample
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as tape:
        # to get a gradient w.r.t x_hat, we need to record the value on the tape
        tape.watch(x_hat)
        out_src, _ = discriminator(x_hat, training=True)
    
    gradients = tape.gradient(out_src, x_hat)
    l2_norm = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
    gp_loss = tf.reduce_mean((l2_norm - 1.0) ** 2)
    return gp_loss


def get_classification_loss(target, logits):
    target = tf.cast(target, dtype=tf.float32)
    logits = tf.squeeze(logits)
    # Compute binary or softmax cross entropy loss.
    loss_total = tf.keras.losses.BinaryCrossentropy(from_logits=True)(target, 
                                                                      logits)
    loss = tf.reduce_mean(loss_total)
    return loss


def get_conditional_expression_loss(true, pred):
    loss_total = tf.keras.losses.MSE(true, pred)
    loss = tf.reduce_mean(loss_total)
    
    return loss


def get_mean_for_loss(out_src):
    return tf.reduce_mean(out_src)


def get_l1_loss(x_real, x_rec):
    return tf.reduce_mean(tf.abs(x_real - x_rec))


def get_attention_loss(fake_attn_mask, rec_attn_mask):
    g_fake_attn_mask_loss = tf.reduce_mean(fake_attn_mask)
    g_rec_attn_mask_loss = tf.reduce_mean(rec_attn_mask)
    g_fake_tv_loss = compute_total_variation_regularization(fake_attn_mask)
    g_rec_tv_loss = compute_total_variation_regularization(rec_attn_mask)
    return g_fake_attn_mask_loss, g_rec_attn_mask_loss, g_fake_tv_loss, g_rec_tv_loss


def compute_total_variation_regularization(attn_mask):
    tv_loss_h = tf.reduce_sum(tf.abs(attn_mask[:, :-1, :, :] - attn_mask[:, 1:, :, :]))
    tv_loss_w = tf.reduce_sum(tf.abs(attn_mask[:, :, :-1, :] - attn_mask[:, :, 1:, :]))
    return tv_loss_h + tv_loss_w