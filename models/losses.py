import tensorflow as tf


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