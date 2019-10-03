import random

import cv2
import numpy as np
import glob


class Dataset:
    def __init__(self, my_dir, my_type="jpg"):
        self.idx = 0
        self.filenames = []
        if my_dir is not None:
            self.load_dir(my_dir, my_type)

    def load_dir(self, my_dir, my_type):
        for file in glob.glob(my_dir + "/*." + my_type):
            self.filenames.append(file)
        return self

    def next(self, batch_size=1):
        batch = []
        i = 0
        while i < batch_size:
            try:
                img = cv2.imread(self.filenames[self.idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch.append(img)
                i += 1
            except Exception as e:
                print(e)
                print("error while reading", self.filenames[self.idx])
            finally:
                self.idx += 1
                if self.idx >= len(self.filenames) - 1:
                    self.shuffle()
                    self.idx = 0

        return np.asarray(batch)

    def shuffle(self):
        random.shuffle(self.filenames)
        return self


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
