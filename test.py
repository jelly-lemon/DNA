import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from train import rename_dir
from util import create_dir

def test(save_dir):
    rename_dir(save_dir, "testMode", 999, 0.99)

if __name__ == '__main__':

    save_dir = create_dir()

    test(save_dir)


