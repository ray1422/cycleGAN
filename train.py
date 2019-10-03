#!/usr/bin/env python3.6
import numpy as np
import tensorflow as tf

import nets
from dataset import Dataset, ImagePool

REAL_LABEL = .9
REAL_LABEL = .9
LEARNING_RATE = 2e-4


def main():
    x_ph = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    y_ph = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])

    fake_x_ph = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
    fake_y_ph = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])

    x = x_ph / 127.5 - 1
    x = tf.image.random_flip_left_right(tf.image.random_flip_up_down(x))
    y = y_ph / 127.5 - 1
    y = tf.image.random_flip_left_right(tf.image.random_flip_up_down(y))

    global_step = tf.Variable(0, trainable=False, dtype=tf.float32)
    global_step_f = tf.Variable(0, trainable=False, dtype=tf.float32)
    global_step_dx = tf.Variable(0, trainable=False, dtype=tf.float32)
    global_step_dy = tf.Variable(0, trainable=False, dtype=tf.float32)

    fake_y = nets.generator(x, scope="G")
    fake_x = nets.generator(y, scope="F")

    rect_x = nets.generator(fake_y, scope="F")
    rect_y = nets.generator(fake_x, scope="G")

    x_disc = nets.discriminator(x, "D_x")
    fake_x_disc = nets.discriminator(fake_x, "D_x")
    fake_x_ph_disc = nets.discriminator(fake_x_ph, "D_x")

    y_disc = nets.discriminator(y, "D_y")
    fake_y_disc = nets.discriminator(fake_y, "D_y")
    fake_y_ph_disc = nets.discriminator(fake_y_ph, "D_y")

    d_x_loss = tf.reduce_mean(tf.square(fake_x_ph_disc)) + tf.reduce_mean(tf.squared_difference(x_disc, REAL_LABEL))
    d_y_loss = tf.reduce_mean(tf.square(fake_y_ph_disc)) + tf.reduce_mean(tf.squared_difference(y_disc, REAL_LABEL))

    g_gan_loss = tf.reduce_mean(tf.squared_difference(fake_y_disc, REAL_LABEL))
    f_gan_loss = tf.reduce_mean(tf.squared_difference(fake_x_disc, REAL_LABEL))

    color_regularize_loss = (tf.reduce_mean(tf.abs(fake_y - x)) + tf.reduce_mean(tf.abs(fake_x - y))) * 2

    cycle_consistent_loss = (tf.reduce_mean(tf.square(rect_x - x)) + tf.reduce_mean(tf.square(rect_y - y))) * 20

    g_loss = cycle_consistent_loss + g_gan_loss + color_regularize_loss
    # g_loss = color_regularize_loss * 2
    f_loss = cycle_consistent_loss + f_gan_loss + color_regularize_loss
    # f_loss = color_regularize_loss * 2

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
    # tf.summary.image("fake/fake_x_ph", _restore_image(fake_x_ph))
    # tf.summary.image("fake/fake_y_ph", _restore_image(fake_y_ph))

    learning_rate = tf.nn.relu(LEARNING_RATE - tf.nn.relu((1 / 100000) * global_step - 100000) * LEARNING_RATE)
    tf.summary.scalar("LR", learning_rate)

    summary = tf.summary.merge_all()

    # optimizers

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

        g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(g_loss,
                                                                                      var_list=tf.trainable_variables(
                                                                                          scope="G"),
                                                                                      global_step=global_step)
        f_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(f_loss,
                                                                                      var_list=tf.trainable_variables(
                                                                                          scope="F"),
                                                                                      global_step=global_step_f)
        d_x_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_x_loss,
                                                                                        var_list=tf.trainable_variables(
                                                                                            scope="D_x"),
                                                                                        global_step=global_step_dx)
        d_y_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(d_y_loss,
                                                                                        var_list=tf.trainable_variables(
                                                                                            scope="D_y"),
                                                                                        global_step=global_step_dy)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(
            var_list=tf.trainable_variables().append(global_step),
            max_to_keep=30)
        ckpt = tf.train.latest_checkpoint("checkpoints")
        if ckpt:
            saver.restore(sess, ckpt)
            print("ckpt loaded!!", ckpt)
        x_dataset = Dataset("datasets/htc")
        y_dataset = Dataset("datasets/ffd")
        x_pool = ImagePool(50)
        y_pool = ImagePool(50)
        train_fake_x = train_fake_y = np.zeros((1, 256, 256, 3))
        step = 0
        train_writer = tf.summary.FileWriter(logdir="logs", graph=sess.graph)
        for i in range(200000):

            if step < 1190 and False:
                pass
                # _, _, _, _, train_fake_x, train_fake_y, step = sess.run(
                #     [g_pre_optimizer, f_pre_optimizer, d_x_optimizer, d_y_optimizer, fake_x, fake_y, global_step],
                #     feed_dict={
                #         x_ph: x_dataset.next(),
                #         y_ph: y_dataset.next(),
                #         fake_x_ph: x_pool.query(train_fake_x),
                #         fake_y_ph: y_pool.query(train_fake_y),
                #     })
                # print("\r[pre: %d]" % step, end="")
            else:
                x_next, y_next = x_dataset.next(), y_dataset.next()
                _, _, _, _, train_summary, step, train_fake_x, train_fake_y = sess.run(
                    [g_optimizer, f_optimizer, d_x_optimizer, d_y_optimizer, summary, global_step, fake_x, fake_y],
                    feed_dict={
                        x_ph: x_next,
                        y_ph: y_next,
                        fake_x_ph: x_pool.query(train_fake_x),
                        fake_y_ph: y_pool.query(train_fake_y),
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
