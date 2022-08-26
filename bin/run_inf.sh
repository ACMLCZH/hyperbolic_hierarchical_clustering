#!/usr/bin/env bash

set -exu

input=$1
timestamp=$2

python -m ghhc.inference.run_inference $input $timestamp