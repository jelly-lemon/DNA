import datetime
import os
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from read_protein import to_number_array
from train import rename_dir
from util import create_dir

def test(save_dir):
    rename_dir(save_dir, "testMode", 999, 0.99)

if __name__ == '__main__':
    r1 = to_number_array('ACDEF')
    print(r1)
    r2 = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    a = np.array([1,2,3,4,5], np.uint8)
    b = np.array([1,2,3,4,5], np.uint8)
    if (a == r2).all():
        print("yes")



