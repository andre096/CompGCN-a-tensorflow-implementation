import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, triples, params, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.triples = triples
        self.shuffle = shuffle
        self.p = params
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.triples) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_tmp = []
        for i in indexes:
            ele = self.triples[i]
            list_tmp.append(self.__data_generator(ele))
        triple = np.asarray([_[0] for _ in list_tmp])
        trp_label = np.asarray([_[1] for _ in list_tmp])
        return (triple[:,0],triple[:,1],triple[:,2]), trp_label

    def __data_generator(self, ele):
        triple, label = np.int32(ele['triple']), np.int32(ele['label'])
        trp_label = self.get_label(label)
        return triple, trp_label

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.triples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_label(self, label):
        """
        Get label corresponding to a (sub, rel) pair.

        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a numpy of shape [num_ent]
        """
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return y
