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
        for i in range(batch_size):
            img = cv2.imread(self.filenames[self.idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch.append(img)
            self.idx += 1
            self.idx %= len(self.filenames)

        return np.asarray(batch)

