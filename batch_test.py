#!/usr/bin/env python3

import argparse
import time
import glob
import tensorflow as tf
import os
import numpy as np
import imageio
import cv2
import multiprocessing as mp

parser = argparse.ArgumentParser(description="batch testing cli tool.")

parser.add_argument('-m', '--model', dest='model', default='pretrained/x2y.pb', help='model path (.pb)')
parser.add_argument('-i', '--input_dir', dest='input_dir', default='samples/origin', help='dir of samples')
parser.add_argument('-o', '--output_dir', dest='output', default='samples/output', help='dir of output')
parser.add_argument('-t', '--threads', dest='threads', default=1, help='num of threads. recommend 1.')
parser.add_argument('--denoise', dest='denoise', action='store_true', help='denoise before sending into generator')
parser.add_argument('--denoise_after', help='denoise after generation', action='store_true')
parser.add_argument('--cpu', help='use cpu only', action='store_true')
parser.add_argument('--debug', help='show all the info.', action='store_true')

FLAGS = parser.parse_args()


def batch_test(files):
    chunks = chunk_it(files, FLAGS.threads)
    threads = []
    lock = mp.Lock()
    num_processed = mp.Value('i', 0)


    try:
        for chunk in chunks:
            if len(chunk) != 0:
                thread = mp.Process(target=process, args=(chunk, lock, num_processed))
                thread.start()
                threads.append(thread)
        i = 0
        print("preparing...")
        time.sleep(3.2)  # 人工等待

        while any(t.is_alive() for t in threads):

            time.sleep(.041666667)  # 24 fps
            i += 1

        print("\nFinish! %d photos has been enhanced." % num_processed.value)

    except KeyboardInterrupt:
        for thread in threads:
            thread.terminate()
            print("process has been killed.")
    finally:
        print()


def process(files, lock, num_processed):
    graph = tf.Graph()
    with graph.as_default():
        image_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 3), name="ph")
        height = tf.placeholder(dtype=tf.int32)
        width = tf.placeholder(dtype=tf.int32)
        input_image = to_float(image_ph)
        with tf.gfile.GFile(FLAGS.model, 'rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
        output_image = tf.import_graph_def(graph_def,
                                           input_map={'input_image': input_image},
                                           return_elements=['output_image:0'],
                                           name='output')
    with tf.Session(graph=graph) as sess:
        i = 0
        num_total = len(files)
        for file in sorted(files):
            try:
                t_start = time.time()
                image = imageio.imread(file)
                h, w, _ = image.shape

                if FLAGS.denoise:
                    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 5, 5)

                image = image[h % 4:, w % 4:, :]
                h, w, _ = image.shape

                [[[result_img]]] = sess.run([output_image], feed_dict={
                    image_ph: image,
                    height: h,
                    width: w
                })
                result_img = np.asarray(result_img)
                if not os.path.isdir(FLAGS.output):
                    os.mkdir(FLAGS.output)

                if FLAGS.denoise_after:
                    result_img = cv2.fastNlMeansDenoisingColored(result_img, None, 10, 10, 5, 5)

                imageio.imwrite(FLAGS.output + "/" + os.path.basename(file), result_img)
                t_now = time.time()
                cost_time = t_now - t_start
                # print("\r" + os.path.basename(file), " %.2fs" % cost_time, " " * 60)
                print("%d / %d" % (i, num_total), end="\r")
                i += 1
                lock.acquire()
                num_processed.value += 1
                lock.release()

            except Exception as e:
                print("Something went wrong!")
                print(str(e))


def main():
    if FLAGS.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if not FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    batch_test(glob.glob(FLAGS.input_dir + "/*.jpg"))


def to_int(image):
    """ cast from float tensor ([-1.,1.]) to int image ([0,255])
    """
    image = tf.clip_by_value(image, -1, 1)
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)


def to_float(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == '__main__':
    main()
