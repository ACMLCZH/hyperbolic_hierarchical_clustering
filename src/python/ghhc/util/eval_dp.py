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

import os
from absl import logging


def eval_dp(filename, outfile, threads, points_file, model_name='ghhc', dataset_name='dataset'):
    """Evaluate dendrogram purity with shell script using xcluster DP code."""

    # outfile = outfile.replace("\\", "/")
    # outfile = os.path.join("C:/Users/v-zhehchen/Documents/GitHub/hyperbolic_hierarchical_clustering", outfile)
    # outfile = "C:/Users/v-zhehchen/Documents/GitHub/hyperbolic_hierarchical_clustering/exp_out/glass/ghhc/" + \
    #           "2022-08-26-11-36-27-ALG=ghhc-IM=randompts-LR=0.01-L=sigmoid-LCA=conditional-NS=50000-BS=500-SP=pcn/1.txt"
    logging.info(outfile)
    os.system("sh bin/score_tree.sh {} {} {} {} {} > {}"
              .format(filename, model_name, dataset_name, threads, points_file, outfile))

    logging.info("???????")

    cost = None
    with open(outfile, 'r') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            cost = float(splt[-1])
    return cost
