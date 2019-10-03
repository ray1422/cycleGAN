#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
import time

import tensorflow as tf

import nets

parser = argparse.ArgumentParser(
    description="This script can export .pb model from checkpoints.",
    formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--ckpt', help='checkpoints directory path', default='checkpoints')
parser.add_argument('-d', '--export_dir', help='dir to save models.', default='pretrained')
parser.add_argument('-f', '--x2y', help='x to y model filename. Please DO NOT included dir.', default='x2y.pb')
parser.add_argument('-b', '--y2x', help='y to x model filename. Please DO NOT included dir.', default='y2x.pb')
parser.add_argument('-s', '--image_size', help='image size', default=396)
parser.add_argument('--gpu', help='using gpu.', action='store_true')
parser.add_argument('--debug', help='show all the info.', action='store_true')

FLAGS = parser.parse_args()


def export_graph():
    graph = tf.Graph()

    with graph.as_default():

        input_image = tf.placeholder(tf.float32, shape=[None, None, 3], name='input_image')

        # cycle_gan.model()
        x2y_images = nets.restore_image(nets.generator(tf.expand_dims(input_image, 0), scope="G"))
        y2x_images = nets.restore_image(nets.generator(tf.expand_dims(input_image, 0), scope="F"))

        thread_x2y = mp.Process(target=export_graph_sess, args=(graph, x2y_images, FLAGS.x2y))
        thread_y2x = mp.Process(target=export_graph_sess, args=(graph, y2x_images, FLAGS.y2x))

        thread_x2y.start()
        if FLAGS.gpu:
            thread_x2y.join()  # or it might out of memory.

        thread_y2x.start()
        i = 0
        while thread_y2x.is_alive() and thread_x2y.is_alive():
            time.sleep(.04166666667)  # ~24 fps
            i += 1
        print("")


def export_graph_sess(graph, result_node, filename):
    output_image = tf.identity(result_node, name='output_image')
    restore_saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def, FLAGS.export_dir, filename, as_text=False)


def main():
    if not FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if not FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ckpt = tf.train.latest_checkpoint(FLAGS.ckpt)
    if not ckpt:
        print("Can't find checkpoint! Did you set --ckpt correctly?")
        exit(1)
    export_graph()


if __name__ == '__main__':
    main()
