"""
Copyright (C) 2019 Authors of gHHC
This file is part of "hyperbolic_hierarchical_clustering"
http://github.com/nmonath/hyperbolic_hierarchical_clustering
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
from absl import logging

tf.enable_eager_execution()


def load_xcluster(filename):
    raw = np.loadtxt(filename, dtype=np.float32)
    pids = raw[:, 0].astype(np.int32)
    lbls = raw[:, 1].astype(np.int32)
    X = raw[:, 2:]
    return pids, lbls, X


def zero_meaned(X):
    return X - np.mean(X, 0)


def unit_normed(X, norm=1.0):
    un = X / np.linalg.norm(X, axis=1, keepdims=True)
    un = tf.clip_by_norm(un, norm, axes=[1]).numpy()
    return un


def load(filename, config, no_lbl=False):
    logging.info('Loading data from filename %s' % filename)
    logging.info('Using xcluster format')
    if no_lbl:
        pids, lbls = None, None
        X = np.load(filename)
    else:
        pids, lbls, X = load_xcluster(filename)

    logging.info(f"{np.linalg.norm(np.mean(X, axis=0))} (mean before)")
    logging.info(f"{np.max(np.linalg.norm(X, axis=1))} (max before)")
    logging.info(f"{np.min(np.linalg.norm(X, axis=1))} (min before)")

    if config.zero_mean:
        logging.info('Zero meaning data')
        X = zero_meaned(X)
    if config.unit_norm:
        logging.info('Unit norming data')
        X = unit_normed(X, config.max_norm)

    logging.info(f"{np.linalg.norm(np.mean(X, axis=0))} (mean after)")
    logging.info(f"{np.max(np.linalg.norm(X, axis=1))} (max after)")
    logging.info(f"{np.min(np.linalg.norm(X, axis=1))} (min after)")

    return pids, lbls, X
