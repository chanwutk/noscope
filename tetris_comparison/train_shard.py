"""Partition the released NoScope training grid without changing its model code."""
import os
import runpy
import sys

import numpy as np
import tensorflow as tf
from keras import backend as keras
from noscope import Models


def main():
    shard = int(os.environ['NOSCOPE_SHARD'])
    workers = int(os.environ['NOSCOPE_WORKERS'])
    assert 0 <= shard < workers
    # Leave enough device memory for the legacy cuDNN handle to initialize.
    keras.set_session(tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=2,
    )))
    # Keep NoScope's random split identical across all workers.
    np.random.seed(0)
    tf.set_random_seed(0)

    upstream_try_params = Models.try_params

    def fit_shard(model_gen, params, *args, **kwargs):
        assigned = params[shard::workers]
        print('Shard {} of {} will fit {}'.format(shard, workers, [p[2:] for p in assigned]))
        return upstream_try_params(model_gen, assigned, *args, **kwargs)

    Models.try_params = fit_shard
    sys.argv[0] = '/noscope/exp/shuffled_small_cnn.py'
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
