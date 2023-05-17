import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
import inspect
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import tensorflow as tf
from tensorflow import keras

np.set_printoptions(precision=4)


def set_gpu(gpus):
    """
    Sets the GPU to be used for the run
    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def ccorr(a,b):
    a = tf.cast(a, tf.complex64)
    b = tf.cast(b, tf.complex64)
    return tf.math.real(tf.signal.ifft(tf.math.conj(tf.signal.fft(a)) * tf.signal.fft(b)))
