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

import sys
import os
import datetime
import numpy as np
import shutil

from absl import logging
import tensorflow as tf

from ghhc.util.Config import Config
from ghhc.util.load import load
from ghhc.util.initializers import random_pts, init_from_afkmc2_and_hac, init_from_rand_and_hac
from ghhc.model.ghhc import gHHCTree, gHHCInference
from ghhc.util.io import mkdir_p

tf.enable_eager_execution()

logging.set_verbosity(logging.INFO)

if __name__ == "__main__":

    config = Config(sys.argv[1])
    ts = sys.argv[2]
    np.random.seed(config.random_seed)

    # now = datetime.datetime.now()
    # ts = "{:04d}{:02d}{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute)
    config.exp_out_dir = os.path.join(config.exp_out_base, config.dataset_name, config.alg_name, ts)
    # "%s-%s" % (, config.to_filename()))
    config.checkpoint_dir = os.path.join(config.exp_out_dir, 'models', 'ckpt')
    if os.path.exists(config.exp_out_dir):
        shutil.rmtree(config.exp_out_dir)
    mkdir_p(config.exp_out_dir)
    mkdir_p(os.path.join(config.exp_out_dir, 'models'))
    config.save_config(config.exp_out_dir, config.to_filename() + ".json")
    config.save_config(config.exp_out_dir)

    for k, v in config.__dict__.items():
        logging.info(f"{k}\t:\t{v}")
    pids, lbls, dataset = load(config.inference_file, config, True)
    dev_pids, dev_lbls, dev_dataset = load(config.dev_file, config, True)
    if config.strict_data is not None:
        dataset = dataset[: config.strict_data]
        dev_dataset = dev_dataset[: config.strict_data]
        if dev_lbls is not None:
            dev_lbls = dev_lbls[: config.strict_data]
    logging.info(f"datasets: {dataset.shape}, {dev_dataset.shape}")
    samples = np.load(config.sample_file)
    logging.info(f"samples: {samples.shape}")

    if config.random_projection is not None:
        logging.info('Using random projection: %s', config.random_projection)
        _proj = np.random.randn(dataset.shape[1], config.random_projection).astype(np.float32)

        def p(x):
            projd = tf.matmul(x, _proj)
            projd /= tf.linalg.norm(projd, axis=1, keepdims=True)
            projd = tf.clip_by_norm(projd, 0.9, axes=[1])
            return projd

        proj = lambda x: p(x)
        init_tree = random_pts(proj(dataset).numpy(), config.num_internals, config.random_pts_scale)
    else:
        if config.init_method == 'randompts':
            init_tree = random_pts(dataset, config.num_internals, config.random_pts_scale)
        elif config.init_method == 'afkmc2hac':
            init_tree = init_from_afkmc2_and_hac(dataset, config.num_internals)
        elif config.init_method == 'randhac':
            init_tree = init_from_rand_and_hac(dataset, config.num_internals, config.random_pts_scale)
        else:
            init_tree = np.full((config.num_internals, dataset.shape[1]), 0.1, dtype=np.float32)
        proj = None

    tree = gHHCTree(init_tree.copy(), config=config, projection=proj)
    optimizer = tf.train.GradientDescentOptimizer(config.tree_learning_rate)
    inf = gHHCInference(tree, optimizer, config, dev_dataset, dev_lbls)

    inf.inference(samples, dataset, config.batch_size)
