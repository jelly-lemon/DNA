import datetime
import os
import time
from collections import Counter

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from read_protein import to_number_array
from train import rename_dir
from util import create_dir
import _pickle as cPickle


def test(save_dir):
    rename_dir(save_dir, "testMode", 999, 0.99)

if __name__ == '__main__':
    with open("./data/protein_sequence/equal/positive_fixed_onehot.pkl")





