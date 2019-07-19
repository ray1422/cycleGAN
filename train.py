#!/usr/bin/env python3.6

import tensorflow as tf

import nets
from dataset import Dataset

REAL_LABEL = .9
LEARNING_RATE = 2e-4


def main():
    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    x = x_ph / 127.5 - 1
    y = y_ph / 127.5 - 1

    global_step = tf.Variable(0, trainable=False, dtype=tf.float32)

    fake_y = nets.generator(x, scope="G")
    fake_x = nets.generator(y, scope="F")
    rect_x = nets.generator(fake_y, scope="F")
    rect_y = nets.generator(fake_x, scope="G")

    x_disc = nets.discriminator(x, "D_x")
    fake_x_disc = nets.discriminator(fake_x, "D_x")

    y_disc = nets.discriminator(y, "D_y")
    fake_y_disc = nets.discriminator(fake_y, "D_y")

    d_x_loss = tf.reduce_mean(tf.square(fake_x_disc)) + tf.reduce_mean(tf.squared_difference(x_disc, REAL_LABEL))
    d_y_loss = tf.reduce_mean(tf.square(fake_y_disc)) + tf.reduce_mean(tf.squared_difference(y_disc, REAL_LABEL))

    g_gan_loss = tf.reduce_mean(tf.squared_difference(fake_y_disc, REAL_LABEL))
    f_gan_loss = tf.reduce_mean(tf.squared_difference(fake_x_disc, REAL_LABEL))

    color_regularize_loss = (tf.reduce_mean(tf.abs(fake_y - x)) + tf.reduce_mean(tf.abs(fake_x - y))) * 2

    cycle_consistent_loss = (tf.reduce_mean(tf.abs(rect_x - x)) + tf.reduce_mean(tf.abs(rect_y - y)))

    g_loss = cycle_consistent_loss * 20 + g_gan_loss + color_regularize_loss
    f_loss = cycle_consistent_loss * 20 + f_gan_loss + color_regularize_loss

    # summary

    tf.summary.histogram("Dx/real", x_disc)
    tf.summary.histogram("Dx/fake", fake_x_disc)
    tf.summary.histogram("Dy/real", y_disc)
    tf.summary.histogram("Dy/fake", fake_y_disc)

    with tf.name_scope("loss"):
        tf.summary.scalar("cycle_consistent", cycle_consistent_loss)
        tf.summary.scalar("color_regularize", color_regularize_loss)
        tf.summary.scalar("dx", d_x_loss)
        tf.summary.scalar("dy", d_y_loss)
        tf.summary.scalar("g_gan", g_gan_loss)
        tf.summary.scalar("f_gan", f_gan_loss)

    with tf.name_scope("X_to_Y_to_X"):
        tf.summary.image("_input", _restore_image(x))
        tf.summary.image("fake_y", _restore_image(fake_y))
        tf.summary.image("reconstruction_x", _restore_image(rect_x))

    with tf.name_scope("Y_to_X_to_Y"):
        tf.summary.image("_input", _restore_image(y))
        tf.summary.image("fake_x", _restore_image(fake_x))
        tf.summary.image("reconstruction_y", _restore_image(rect_y))

    learning_rate = tf.nn.relu(LEARNING_RATE - tf.nn.relu((1 / 100000) * global_step - 100000) * LEARNING_RATE)
    tf.summary.scalar("LR", learning_rate)

    summary = tf.summary.merge_all()

    # optimizers
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_optimizer = optimizer.minimize(g_loss, var_list=tf.trainable_variables(scope="G"), global_step=global_step)
    f_optimizer = optimizer.minimize(f_loss, var_list=tf.trainable_variables(scope="F"))
    d_x_optimizer = optimizer.minimize(d_x_loss, var_list=tf.trainable_variables(scope="D_x"))
    d_y_optimizer = optimizer.minimize(d_y_loss, var_list=tf.trainable_variables(scope="D_y"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.trainable_variables().append(global_step), max_to_keep=10)
        ckpt = tf.train.latest_checkpoint("checkpoints")
        if ckpt:
            saver.restore(sess, ckpt)
            print("ckpt loaded!!", ckpt)
        x_dataset = Dataset("datasets/noisy")
        y_dataset = Dataset("datasets/clear")
        train_writer = tf.summary.FileWriter(logdir="logs", graph=sess.graph)
        while True:
            _, _, _, _, train_summary, step = sess.run(
                [g_optimizer, f_optimizer, d_x_optimizer, d_y_optimizer, summary, global_step], feed_dict={
                    x_ph: x_dataset.next(),
                    y_ph: y_dataset.next()
                })

            if step % 300 == 0 or (step < 600 and step % 10 == 0):
                train_writer.add_summary(train_summary, global_step=step)
                train_writer.flush()

            if step % 1000 == 1:
                saver.save(sess, "checkpoints/checkpoint.ckpt", global_step=int(step))

            print("\r[step: %d]" % step, end="")


def _restore_image(x):
    return tf.cast(tf.clip_by_value((x + 1) * 127.5, 0, 255), dtype=tf.uint8)


if __name__ == '__main__':
    main()
