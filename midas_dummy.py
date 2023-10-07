#!/usr/bin/env python3

import os
import argparse
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from flopth import flopth

def load_lf(file, device):
    lf = np.load(file)
    #print(lf.shape)
    #img = lf[4, 4, ...]
    img = lf[3, 3, ...]
    img = np.transpose(img, (1, 2, 0))
    #img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img #.to(device)


def infer(args):
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    os.makedirs(args.out_path, exist_ok=True)
    
    f = open(args.out_path + f"{args.dataset}_{args.width}x{args.height}_timing.txt", "w")

    all_times = []
    with torch.no_grad():
        lfs = glob.glob(args.lfs_path + f"{args.dataset}/test/*.npy", recursive=True)
        print(f"Found {len(lfs)} images. Saving files to {args.out_path}/")

        for file in tqdm(lfs):
            img = load_lf(file, device)
            input_batch = transform(img).to(device)
            #print(img.shape, input_batch.shape)

            # flops, params = flopth(midas, inputs=(input_batch,),show_detail=True)
            # print(flops, params)
            # exit()

            starttime = time.time()
            depth = midas(input_batch)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1),
                                                    size=(args.height, args.width),
                                                    mode="bicubic", 
                                                    align_corners=False).squeeze()
            output = depth/60
            time_elapsed = time.time() - starttime
            all_times.append(time_elapsed)
            output = output.cpu().numpy()

            f.write(file + ":" + str(time_elapsed) + "\n")

    f.write("\nAvg time: " + str(sum(all_times)/len(all_times)) + "\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default="Hybrid")
    parser.add_argument('-p', '--lfs_path', help="path to all lfs", default="/media/data/prasan/datasets/LF_datasets/")
    parser.add_argument('-o', '--out_path', help="directory to save output", default="/data2/aryan/mono-eccv/depth-timing/")
    parser.add_argument('-th', '--height', help="height of input", default=176, type=int)
    parser.add_argument('-tw', '--width', help="width of input", default=264, type=int)
    args = parser.parse_args()

    infer(args)