#!/usr/bin/env python3

import sys
import os
import subprocess
import glob
from tqdm.auto import tqdm 

def getResults(dataset, h, w, noRef = False):
    assert dataset is not None, "Please provide dataset name"

    if noRef:
        os.system(f"./run_test.py --gpu_1 3 --gpu_2 3 --calcMetrics \
              --dataset {dataset} \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --genDP_path /data2/aryan/{dataset}/test_dp/ \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              --no_refinement \
              -th {h} -tw {w} -vh {h} -vw {w}")
    else:
        os.system(f"./run_test.py --gpu_1 3 --gpu_2 3 --calcMetrics \
              --dataset {dataset} \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --genDP_path /data2/aryan/{dataset}/test_dp/ \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {h} -tw {w} -vh {h} -vw {w}")


if __name__ == '__main__':
    datasets = ['Hybrid', 'Stanford', 'Kalantari', 'TAMULF']
    # our: 256x192 | Govindrajan: 176x264 | Li: 192x192 | Srinivasan: 188x270 |
    # Random: 384x528 (to introduce another baseline) | 480p-SD: 480x640
    resH = [256, 176, 192, 188, 384, 480]
    resW = [192, 264, 192, 270, 528, 640]
    noRef = [True, False]

    for noRef_bool in noRef:
        for dataset in (pbar:=tqdm(datasets)):
            pbar.set_description(f"Processing {dataset}")
            for i in range(len(resH)):
                getResults(dataset, resH[i], resW[i], noRef=noRef_bool)