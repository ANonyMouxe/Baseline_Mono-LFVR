#!/usr/bin/env python3

import os
from tqdm.auto import tqdm 


def getResults(dataset, h, w, noRef = False):
    assert dataset is not None, "Please provide dataset name"

    if noRef:
        os.system(f"./run_test.py --gpu_1 1 --gpu_2 1 --calcMetrics \
              --dataset {dataset} \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --genDP_path /data2/aryan/{dataset}/test/ \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              --no_refinement \
              -th {h} -tw {w} -vh {h} -vw {w}")
    else:
        os.system(f"./run_test.py --gpu_1 1 --gpu_2 1 --calcMetrics \
              --dataset {dataset} \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --genDP_path /data2/aryan/{dataset}/test/ \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {h} -tw {w} -vh {h} -vw {w}")


def dispMethodComparison(dataset, rH = 352, rW = 528, ref=False): 
    assert dataset is not None, "Please provide dataset name"

    rh, rw = rH, rW 
    print(f"Resolution: {rh} x {rw}")

    disp_paths = ['/data2/aryan/unimatch/dp_otherDS/', '/media/data/prasan/datasets/LF_datasets/DPT-depth/']

    for dips_path in disp_paths:
        if "unimatch" in dips_path:
            print(f"Using Unimatch Disparities")
        else:
            print(f"Using DPT Disparities")

        if ref:
            os.system(f"./run_test.py --calcMetrics \
              --gpu_1 1 --gpu_2 1 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {rh} -vh {rh} \
              -tw {rw} -vw {rw} \
              --otherDS_disp_path {dips_path}")
        else:
            os.system(f"./run_test.py --calcMetrics --no_refinement \
              --gpu_1 1 --gpu_2 1 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {rh} -vh {rh} \
              -tw {rw} -vw {rw} \
              --otherDS_disp_path {dips_path}")


def calc_DPT_Time(dataset, h, w):
    assert dataset is not None, "Please provide dataset name"
    assert h is not None, "Please provide height"
    assert w is not None, "Please provide width"

    os.system(f"./midas_dummy.py --dataset {dataset} -th {h} -tw {w}")


def calcTime(dataset, h, w, no_ref=None):
    assert dataset is not None, "Provide DS"
    assert no_ref is not None, "Provide an inference mode"

    if no_ref:
        os.system(f"./run_test.py --calcMetrics --calcTime --no_refinement \
              --results net_timing \
              --gpu_1 3 --gpu_2 3 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {h} -vh {h} \
              -tw {w} -vw {w} \
              --otherDS_disp_path /media/data/prasan/datasets/LF_datasets/DPT-depth/")
    else:
        os.system(f"./run_test.py --calcMetrics --calcTime \
              --results net_timing \
              --gpu_1 3 --gpu_2 3 \
              --dataset {dataset} \
              --genDP_path /data2/aryan/{dataset}/test_dp \
              --filenames_file_eval ./FT_train-test_files/{dataset}/test_files.txt \
              --filenames_file_folder /data2/aryan/mono-eccv/FT_train-test_files/{dataset}/ \
              -th {h} -vh {h} \
              -tw {w} -vw {w} \
              --otherDS_disp_path /media/data/prasan/datasets/LF_datasets/DPT-depth/")


if __name__ == '__main__':
    datasets = ['Hybrid', 'Stanford', 'Kalantari', 'TAMULF']
    # our: 256x192 | Govindrajan: 176x264 | Li: 192x192 | Srinivasan: 188x270 |
    # Random: 384x528 (to introduce another baseline) | 480p-SD: 480x640
    resH = [256, 176, 192, 384, 480]
    resW = [192, 264, 192, 528, 640]
    noRef = [True, False]

    # for noRef_bool in noRef:
    #     for dataset in (pbar:=tqdm(datasets)):
    #         pbar.set_description(f"Processing {dataset}")
    #         dispMethodComparison(dataset)
    #         # for i in range(len(resH)):
    #         #     getResults(dataset, resH[i], resW[i], noRef=noRef_bool)

    resH2 = [352]
    resW2 = [528]

    for dataset in datasets:
        for i in range(len(resH)):
            dispMethodComparison(dataset, resH[i], resW[i], ref=False)
            dispMethodComparison(dataset, resH[i], resW[i], ref=True)
            # print(f"DPT Timing on {dataset}")
            # calc_DPT_Time(dataset, resH2[i], resW2[i])
            # print(f"Network Timing on no_ref {dataset}")
            # calcTime(dataset, resH2[i], resW2[i], no_ref=True)
            # print(f"Network Timing (+ Ref) {dataset}")
            # calcTime(dataset, resH2[i], resW2[i], no_ref=False)