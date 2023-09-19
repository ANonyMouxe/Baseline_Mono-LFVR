#!/usr/bin/env python
import argparse
import os
import sys
import uuid
from datetime import datetime as dt
import json
import numpy as np
import random
import math
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torchvision
from lpips_pytorch import LPIPS

import model_io
import models
from vid_dataloader import LFDataLoader
from loss import *
from utils import RunningAverage, RunningAverageDict, denormalize
import tensor_ops as utils


class Tester():
    def __init__(self, args):

        self.args = args
        #################################### Setup GPU device ######################################### 
        self.device = torch.device(f'cuda:{args.gpu_1}' if torch.cuda.is_available() else 'cpu')
        self.device1 = torch.device(f'cuda:{args.gpu_2}' if torch.cuda.is_available() else 'cpu')
        print('Device: {}, {}'.format(self.device, self.device1))
        
        self.no_refinement = args.no_refinement
        #################################### Refinement Model #########################################
        if not self.no_refinement:
            self.ref_model = models.RefinementBlock(patch_size=1)
            checkpoint = torch.load('weights/eccv_refine_net.pt', map_location='cpu')['model']
            self.ref_model.load_state_dict(checkpoint)
            self.ref_model = self.ref_model.to(self.device)

        ####################################### LF Model ##############################################
        # number of predictions for TD
        td_chans = self.args.rank*self.args.num_layers*3
        self.model = models.UnetLF.build(td_chans=td_chans, layers=args.num_layers, rank=args.rank)
        self.model.encoder.original_model.conv_stem = models.Conv2dSame(10, 48, kernel_size=(3, 3), 
                                                                           stride=(2, 2), bias=False)
        checkpoint = torch.load('weights/eccv_recons_net.pt', map_location='cpu')['model']
        self.model.load_state_dict(checkpoint)
        # print(self.lf_model)
        self.model = self.model.to(self.device1)

        ##################################### Tensor Display ##########################################
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device1)
        self.md = args.max_displacement
        self.zp = args.zero_plane

        ####################################### Save test results ##############################################
        self.save_path = os.path.join(args.results, "Ref_" + str(not args.no_refinement) + f"_{args.val_height}x{args.val_width}_" + args.dataset)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_numpy = args.save_numpy


    def calculate_psnr(self, img1, img2):
        # print(img1.shape, img2.shape)
        with torch.no_grad():
            img1 = 255*img1#.cpu()
            img2 = 255*img2#.cpu()
            mse = torch.mean((img1 - img2)**2)
            if mse == 0:
                return float('inf')
            return 20 * math.log10(255.0 / math.sqrt(mse))


    def calculate_ssim(self, img1, img2):
        with torch.no_grad():
            ssim = SSIMLoss()
            N, V, C, H, W = img1.shape
            img1 = img1.reshape(N*V, C, H, W).cpu()
            img2 = img2.reshape(N*V, C, H, W).cpu()
            loss = 1 - 0.1*ssim(img1, img2)
            return loss


    def test(self, test_loader, max_disp, zero_plane):
        ###############################################################################################
        # some globals
        iters = len(test_loader)
        # interpolate = nn.Upsample(size=(1080, 720), mode='bilinear')
        if self.args.calcMetrics:
            psnr_avg_1 = RunningAverage()
            ssim_avg_1 = RunningAverage()
            f = open(os.path.join(self.save_path, f'results.txt'), 'w')

        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing {self.args.dataset}") as vepoch:
                for i, batch in vepoch:
                    # print(i, batch.keys()) 
                    video_length = len(batch)
                    prev_state = None

                    # pred_lfs = []
                    # orig_imgs = []
                    
                    if self.args.calcMetrics:
                        # ids = sorted(batch.keys())
                        # leng = len(ids) - 1
                        gt_lfs = []
                        psnrs_1 = []
                        ssims_1 = []
                        curr_img = batch[1]['rgb'].to(self.device)
                        prev_img = batch[0]['rgb'].to(self.device)
                        next_img = batch[2]['rgb'].to(self.device)
                        # NOTE: prev and next will be used for video input methods
                        curr_orig_image = batch[1]['rgb']
                        # orig_imgs.append(denormalize3d(curr_orig_image).cpu())
                        # prev_orig_image = batch[id-1]['rgb']
                        # next_orig_image = batch[id+1]['rgb']
                        unimatch_disp =  batch[1]['disp'].to(self.device)
                        disp = -1 * (unimatch_disp - zero_plane) * max_disp
                        # print(disp.shape)
                        img = torch.cat([prev_img, curr_img, next_img, disp], dim=1) 
    
                        decomposition, depth_planes, state = self.model(img, prev_state)
                        pred_lf = self.val_td_model(decomposition, depth_planes).cpu()
                        pred_lf.clip(0, 1)
                        # pred_lfs.append(pred_lf)
                        if self.args.calcMetrics:
                            # print(f"calculating metrics for {id}th image")
                            gt_lf = batch[1]['lf'].cpu()
                            # gt_lfs.append(gt_lf)
                            pred_psnr_1 = self.calculate_psnr(pred_lf, gt_lf)
                            pred_ssim_1 = self.calculate_ssim(pred_lf, gt_lf)
                            # print("psnr, ssim :: ", pred_psnr_1, pred_ssim_1)
                            psnr_avg_1.append(pred_psnr_1)
                            ssim_avg_1.append(pred_ssim_1)
                            psnrs_1.append(pred_psnr_1) # running avg
                            ssims_1.append(pred_ssim_1)
                    
                        avg_psnr_1 = sum(psnrs_1)/len(psnrs_1)
                        avg_ssim_1 = sum(ssims_1)/len(ssims_1)
                        string = 'Sample {0:2d} => PSNR: {1:.4f}, SSIM: {2:.4f}\n'.format(i, avg_psnr_1, avg_ssim_1)
                        f.write(string)
                        vepoch.set_postfix(psnr_iA=f"{psnr_avg_1.get_value():0.2f}({avg_psnr_1:0.2f})",
                                       ssim_iA=f"{ssim_avg_1.get_value():0.2f}({avg_ssim_1:0.2f})")

                if self.args.calcMetrics:
                    pred_avg_psnr = psnr_avg_1.get_value()
                    pred_avg_ssim = ssim_avg_1.get_value()
                    string = '\n\n---------\nAverage PSNR: {0:.4f}\nAverage SSIM: {1:.4f}\n---------'.format(pred_avg_psnr, pred_avg_ssim)
                    f.write(string)
                    f.close()

                        

    def main_worker(self):        
        ###############################################################################################
        test_loader = LFDataLoader(self.args, mode='test', calcMetrics=self.args.calcMetrics).data
        # print("test_loader sample:", next(iter(test_loader))[0]['disp'].shape)
        zp, md = self.zp, self.md
        self.test(test_loader, md, zp)



def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Testing script. Default values of all arguments are recommended for reproducibility', 
                                     fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    ####################################### Experiment arguments ######################################
    parser.add_argument('--results', default='results', type=str, help='directory to save results')
    parser.add_argument('-sn', '--save_numpy', default=False, action='store_true', help='whether to save np files')
    parser.add_argument('--mode', default='test', type=str, help='train or test')

    parser.add_argument('--dataset', default='TAMULF', type=str, help='Dataset to test on')
    parser.add_argument('--filenames_file_eval', default='./FT_train-test_files/TAMULF/test_files.txt',
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--calcMetrics', help='if set, calculates metrics', action='store_true', default=True)
    parser.add_argument('--lf_path', default='/data/prasan/datasets/LF_datasets/', type=str, help='path to light field dataset')
    parser.add_argument('-ty', '--type', type=str, default='resize', 
                        help='whether to train with crops or resized images')
    parser.add_argument('--genDP_path', default='/data2/aryan/TAMULF/test_dp/', type=str, help='path to generated dual pixels') 
    
    # Change to DPT depth maps: /data/prasan/datasets/LF_datasets/DPT-depth/ | /data2/aryan/unimatch/dp_otherDS/
    parser.add_argument('--otherDS_disp_path', default='/data2/aryan/unimatch/dp_otherDS/', type=str, help='path to other datasets disparity maps')

    
    parser.add_argument('--gpu_1', default=0, type=int, help='which gpu to use')
    parser.add_argument('--gpu_2', default=0, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=1, type=int, help='number of workers for data loading')

    ######################################## Dataset parameters #######################################

    parser.add_argument('--datapath', default='/data/prasan/datasets/LF_datasets/', type=str,
                        help='path to dataset')
    parser.add_argument('--unrect_datapath', default='/data2/raghav/datasets/Pixel4_3DP/unrectified', type=str,
                        help='path to dataset')
    
    # Unimatch disparity maps ----------------------------------------------------------
    # NOTE: Default: True
    parser.add_argument('--unimatch_disp_path', '-udp', default='/data2/aryan/lfvr/disparity_maps/disp_pixel4_BA', type=str,
                        help='path to disparity maps from unimatch')
    parser.add_argument('--use_unimatch', '-uud', default=True, action='store_true')
    # -----------------------------------------------------------------------------------

    parser.add_argument('--filenames_file_folder',
                        default='/data2/aryan/mono-eccv/FT_train-test_files/TAMULF/',
                        type=str, help='path to the folder containing filenames to use')

    parser.add_argument('--visualization_shuffle', default=False, action='store_true', help='visualize input data')


    # parser.add_argument('-ty', '--type', type=str, default='resize', 
    #                     help='whether to train with crops or resized images')

    ############################################# I/0 parameters ######################################
    parser.add_argument('-th', '--train_height', type=int, help='train height', default=480)
    parser.add_argument('-tw', '--train_width', type=int, help='train width', default=640)
    # HW: mono-lfvr: 176x264 (D) | Srinivasan: 188 x 270 (D) | Li: 192x192 | ours: 256x192 (D) | Random: 352x528 (D) | SD 480p: 480x640
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=480)
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=640)

    parser.add_argument('--depth_input', default=False, action='store_true', 
                        help='whether to use depth as input to network')
    
    parser.add_argument('-md', '--max_displacement', default=1.2, type=float)
    parser.add_argument('-zp', '--zero_plane', default=0.3, type=float)
    parser.add_argument('-cc', '--color_corr', default=True, action='store_true')

    ##################################### Learning parameters #########################################
    parser.add_argument('-bs', '--batchsize', default=1, type=int, help='batch size')

    ##################################### Tensor Display parameters #########################################
    parser.add_argument('--rank', default= 12, type=int, help='rank of the tensor display')
    parser.add_argument('--num_layers', default= 3, type=int, help='number of layers in the tensor display')
    parser.add_argument('--angular', default= 7, type=int, help='number of angular views to output')
    parser.add_argument('-tdf', '--td_factor', default=1, type=int, help='disparity factor for layers')
    # by default always use refinement network
    parser.add_argument('--no_refinement', default=False, action='store_true', help='whether to use refinement network')

    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    tester = Tester(args)
    tester.main_worker()