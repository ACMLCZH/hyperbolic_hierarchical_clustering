import os
import time

import numpy as np
import tensorflow as tf

from ghhc.util.eval_dp import eval_dp
from ghhc.util.io import mkdir_p

from absl import logging

tf.enable_eager_execution()


def squared_norm(x, axis=1, keepdims=True):
    """Squared L2 Norm of x."""
    return tf.reduce_sum(tf.pow(x, 2), axis=axis, keepdims=keepdims)


def squared_euclidean_cdist(x, y):
    """Squared euclidean distance

    Computed as: ||x||^2 + ||y||^2 - 2 x^T y.

    Args:
        x: N by D matrix
        y: M by D matrix
    :returns matrix (N by M) such that result[i,j] = || x[i,:] - y[j,;] ||^2
    """
    norms = squared_norm(x, axis=1, keepdims=True) + tf.transpose(squared_norm(y, axis=1, keepdims=True))
    dot = 2.0 * tf.matmul(x, y, transpose_b=True)
    return norms - dot


def poincare_cdist(x, y):
    """Poincare distance

    Args:
        x: N by D matrix
        y: M by D matrix
    :returns matrix (N by M) such that result[i,j] = ppoincare dist(x[i,:], y[j,:])
    """
    numerator = squared_euclidean_cdist(x, y)
    denom = (1.0 - squared_norm(x)) * (1.0 - tf.transpose(squared_norm(y, axis=1, keepdims=True)))
    arccosh_arg = 1.0 + 2.0 * numerator / denom
    res = tf.math.acosh(1e-8 + arccosh_arg)
    return res


def squared_euclidean_dist(x, y):
    """Squared euclidean distance

    Computed as: ||x||^2 + ||y||^2 - 2 x^T y.

    Args:
        x: N by D matrix
        y: N by D matrix
    :returns vector (N by 1) such that the ith element is || x[i,:] - y[i,;] ||^2
    """
    norms = squared_norm(x, axis=1, keepdims=True) + squared_norm(y, axis=1, keepdims=True)
    dot = 2 * tf.reduce_sum(tf.multiply(x, y), axis=1, keepdims=True)
    return norms - dot


def poincare_dist(x, y):
    """Poincare distance between x and y.

        Args:
            x: N by D matrix
            y: N by D matrix
        :returns vector (N by 1) such that the ith element is poincare dist(x[i,:], y[i,:])
        """
    numerator = squared_euclidean_dist(x, y)
    denom = (1.0 - squared_norm(x)) * (1.0 - squared_norm(y))
    arccosh_arg = 1.0 + 2.0 * numerator / denom
    res = tf.math.acosh(arccosh_arg)
    return res


def poincare_norm(x, axis=1, keepdims=True):
    """Squared poincare norm of x."""
    return 2.0 * tf.math.atanh(tf.linalg.norm(x, axis=axis, keepdims=keepdims))


def parent_order_penalty(p, c, marg):
    """Penalty for parents to have smaller norm than children."""
    return tf.maximum(0.0, poincare_norm(p) - poincare_norm(c) + marg) + 1.0


def parent_order_penalty_cdist(p, c, marg):
    """Penalty for parents to have smaller norm than children."""
    return tf.maximum(0.0, tf.transpose(poincare_norm(p)) - poincare_norm(c) + marg) + 1.0


class TableTree:
    """Object for a table tree."""
    def __init__(self, filename):
        data = np.loadtxt(filename, dtype=np.float32)
        self.nodeVec = data[:, :-1]
        self.nodeFa = data[:, -1].astype(np.int)
        self.nodeSon = list()
        self.numNode = data.shape[0]
        for i in range(self.numNode):
            self.nodeSon.append(list())
            if self.nodeFa[i] != -1:
                self.nodeSon[self.nodeFa[i]].append(i)


def rsgd_or_sgd(grads_and_vars, rsgd=True):
    if rsgd:
        res = []
        for g, v in grads_and_vars:
            scale = ((1.0 - tf.reduce_sum(tf.multiply(v, v), axis=1, keepdims=True)) ** 2) / 4.0
            res.append((scale * g, v))
        return res
    else:
        return grads_and_vars


class gHHCInference(object):
    def __init__(self, ghhcTree, optimizer, config, dev_set, dev_lbls):
        self.ghhcTree = ghhcTree
        self.optimizer = optimizer
        self.config = config
        self.dev_set = dev_set
        self.dev_lbls = dev_lbls
        self.best_dev_dp_score = 0.0
        self.best_dev_iter = 0.0
        self.last_dev_dp_score = 0.0
        self.last_dev_iter = 0.0
        self.checkpoint_prefix = self.config.checkpoint_dir + "/ckpt"
        self.ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                        model=ghhcTree,
                                        optimizer_step=tf.train.get_or_create_global_step())

    def update(self, c1, c2, par_id, gp_id, steps=100):
        for i in range(steps):
            with tf.GradientTape() as tape:
                loss = self.ghhcTree.pull_close_par_gp(c1, c2, par_id, gp_id)
            grads = tape.gradient(loss, self.ghhcTree.trainable_variables)
            self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                           global_step=tf.train.get_or_create_global_step())
            self.ghhcTree.clip()

    def episode_inference(self, x_i, x_j, x_k, dataset, batch_size=1000, examples_so_far=0):
        time_so_far = 0.0
        loss_so_far = 0.0
        struct_loss_so_far = 0.0

        for idx in range(0, x_i.shape[0], batch_size):

            if self.config.struct_prior is not None and idx + examples_so_far > 0:
                if self.ghhcTree.cached_pairs is None:
                    self.dev_eval(idx + examples_so_far)

                # update parameters by structure loss every struct_prior_every steps
                if (idx + examples_so_far) % self.config.struct_prior_every == 0:
                    for idx2 in range(self.config.num_struct_prior_batches):
                        start_time = time.time()
                        logging.log_every_n(
                            logging.INFO, '[STRUCTURE] Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (
                                idx2, 100, struct_loss_so_far / max(idx2, 1), time_so_far / max(idx2, 1)
                            ), 100)
                        with tf.GradientTape() as tape:
                            sloss = self.ghhcTree.structure_loss()
                            struct_loss_so_far += sloss.numpy()
                        grads = tape.gradient(sloss, self.ghhcTree.trainable_variables)
                        self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                                       global_step=tf.train.get_or_create_global_step())
                        self.ghhcTree.clip()
                        end_time = time.time()
                        time_so_far += end_time - start_time
                    logging.log(logging.INFO,
                                '[STRUCTURE] Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (
                                    self.config.num_struct_prior_batches, 100,
                                    struct_loss_so_far / max(self.config.num_struct_prior_batches, 1),
                                    time_so_far / max(self.config.num_struct_prior_batches, 1)))

            if (idx + examples_so_far) % self.config.dev_every == 0:
                self.dev_eval(idx + examples_so_far)
            elif (idx + examples_so_far) % self.config.save_every == 0:
                self.ckpt.save(self.checkpoint_prefix)
                self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
                self.config.save_config(self.config.exp_out_dir, filename='config.json')

            start_time = time.time()
            if idx % 100 == 0 and idx > 0:
                logging.info('Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (
                    idx, x_i.shape[0], loss_so_far / idx, time_so_far / max(idx, 1)))
            with tf.GradientTape() as tape:
                bx_i = dataset[x_i[idx:(idx + batch_size)], :]
                bx_j = dataset[x_j[idx:(idx + batch_size)], :]
                bx_k = dataset[x_k[idx:(idx + batch_size)], :]
                loss = self.ghhcTree.compute_loss(bx_i, bx_j, bx_k)
                loss_so_far += loss.numpy()
            grads = tape.gradient(loss, self.ghhcTree.trainable_variables)
            self.optimizer.apply_gradients(rsgd_or_sgd(zip(grads, self.ghhcTree.trainable_variables)),
                                           global_step=tf.train.get_or_create_global_step())
            self.ghhcTree.clip()
            end_time = time.time()
            time_so_far += end_time - start_time

        logging.info('Processed %s of %s batches || Avg. Loss %s || Avg Time %s' % (
            x_i.shape[0], x_i.shape[0], loss_so_far / x_i.shape[0], time_so_far / max(x_i.shape[0], 1)))

        # save model at the end of training
        self.ckpt.save(self.checkpoint_prefix)
        self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        # record the last model in the config.
        self.config.save_config(self.config.exp_out_dir, filename='config.json')
        return x_i.shape[0]

    def dev_eval(self, steps):
        if self.dev_set is not None:
            start_dev = time.time()
            mkdir_p(os.path.join(self.config.exp_out_dir, 'dev'))
            filename = os.path.join(self.config.exp_out_dir, 'dev', 'dev_tree_%s.tsv' % steps)
            self.ghhcTree.write_tsv(filename, self.dev_set, lbls=self.dev_lbls)
            dp = eval_dp(filename, os.path.join(self.config.exp_out_dir, 'dev', 'dev_score_%s.tsv' % steps),
                         self.config.threads, self.config.dev_points_file)
            logging.info('DEV EVAL @ %s minibatches || %s DP' % (steps, dp))
            end_dev = time.time()
            logging.info('Finished Dev Eval in %s seconds' % (end_dev - start_dev))
            if self.config.save_dev_pics:
                filename = os.path.join(self.config.exp_out_dir, 'dev', 'dev_tree_%s.png' % steps)
                self.ghhcTree.plot_tree(self.dev_set, filename)

            # record the best dev score to try to understand if we end up doing worse, not used at inference time
            # last model is used at inference.
            self.best_dev_dp_score = max(self.best_dev_dp_score, dp)
            self.best_dev_iter = steps if self.best_dev_dp_score == dp else self.best_dev_iter
            self.last_dev_dp_score = dp
            self.last_dev_iter = steps
            # save every time we run this eval
            self.ckpt.save(self.checkpoint_prefix)
            self.config.last_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            if self.best_dev_dp_score == dp:
                self.config.best_model = tf.train.latest_checkpoint(self.config.checkpoint_dir)
            self.config.save_config(self.config.exp_out_dir, filename='config.json')
            return dp
        else:
            return 0.0

    def inference(self, indexes, dataset, batch_size=1000, episode_size=5000):
        batches_so_far = 0
        curr_idx = 0
        episode_size = self.config.episode_size
        if self.config.shuffle:
            indexes = indexes[np.random.permutation(indexes.shape[0]), :]
        for i in range(self.config.num_iterations):
            logging.info(f"curr_idx = {curr_idx}")
            if curr_idx > indexes.shape[0]:
                logging.info('Restarting....')
                curr_idx = 0
                if self.config.shuffle:
                    indexes = indexes[np.random.permutation(indexes.shape[0]), :]
            logging.info('Starting iteration %s of %s' % (i, self.config.num_iterations))
            batches_so_far += self.episode_inference(indexes[curr_idx:(curr_idx + episode_size), 0],
                                                     indexes[curr_idx:(curr_idx + episode_size), 1],
                                                     indexes[curr_idx:(curr_idx + episode_size), 2],
                                                     dataset, batch_size, examples_so_far=batches_so_far)
