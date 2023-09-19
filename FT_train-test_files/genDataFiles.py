#!/usr/bin/env python

import os
import sys
import random

# format: tab separated dataset/test/{}
def main(dataPath, ds):
    with open(f"{ds}/eval_files.txt", "w") as f:
        all_lfs = os.listdir(f"{dataPath}/{ds}/test/")
        for lf in all_lfs:
            f.write(f"{ds}/test/{lf}\t{ds}/test_dp/left_{lf.split('.')[0]}.png\t{ds}/test_dp/right_{lf.split('.')[0]}.png\n")
            

if __name__ == '__main__':
    dataPath = "/data/prasan/datasets/LF_datasets/"
    datasets = ["Stanford", "TAMULF", "Hybrid", "Kalantari"]
    for ds in datasets:
        main(dataPath, ds)
