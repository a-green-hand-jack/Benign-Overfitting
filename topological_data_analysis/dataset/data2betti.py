!pip install ripserplusplus
!pip install ripser

import ripserplusplus as rpp_py
import numpy as np
from tqdm import tqdm
import sys
from ripser import ripser
import time

def distance_betti(distances=None):
    start = time.time()
    num_iters= len(distances)

    d = rpp_py.run("--format distance", distances)

    end = time.time()
    print("ripser++ total time: ", end-start)

    return d