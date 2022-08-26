#!/usr/bin/env bash

set -exu

timestamp=2022$2
data=$1
data_to_run_on=$1


sh bin/sample_triples.sh config/$data/build_samples.json
sh bin/run_inf.sh config/$data/$data.json $timestamp
sh bin/run_predict_only.sh exp_out/$data/ghhc/$timestamp/config.json data/$data/$data_to_run_on.tsv
sh bin/score_tree.sh exp_out/$data/ghhc/$timestamp/results/tree.tsv