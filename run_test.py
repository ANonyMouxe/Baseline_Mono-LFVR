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
        self.lf_model = models.UnetLF.build(td_chans=td_chans, layers=args.num_layers, rank=args.rank)
        self.lf_model.encoder.original_model.conv_stem = models.Conv2dSame(10, 48, kernel_size=(3, 3), 
                                                                           stride=(2, 2), bias=False)
        checkpoint = torch.load('weights/eccv_recons_net.pt', map_location='cpu')['model']
        self.lf_model.load_state_dict(checkpoint)
        # print(self.lf_model)
        self.lf_model = self.lf_model.to(self.device1)

        ##################################### Tensor Display ##########################################
        self.val_td_model = models.multilayer(height=args.val_height, width=args.val_width, 
                                              args=self.args, device=self.device1)
        self.md = args.max_displacement
        self.zp = args.zero_plane

        ####################################### Save test results ##############################################
        self.save_path = os.path.join(args.results, args.dataset+'-{:.2f}, {:.2f}'.format(self.md, self.zp))
        os.makedirs(self.save_path, exist_ok=True)
        self.save_numpy = args.save_numpy


    def calculate_psnr(self, img1, img2):
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


    def test(self, test_loader, md, zp):
        ###############################################################################################
        if not self.no_refinement:
            self.ref_model.eval()
            
        self.lf_model.eval()

        ################################# Validation loop #############################################
        with torch.no_grad():
            with tqdm(enumerate(test_loader), total=len(test_loader), 
                      desc='Testing-{}_{:.2f},{:.2f}'.format(self.args.dataset, md, zp)) as vepoch:
                for i, batch in vepoch:
                    prev_state = None
                    video_length = len(batch)
                    pred_lfs = []
                    if not self.no_refinement:
                        ref_lfs = []
                    # gt_lfs = []
                    orig_imgs = []
                    diff_lfs = []
                    masks = []

                    psnrs_1 = []
                    ssims_1 = []
                    psnrs_2 = []
                    ssims_2 = []

                    for id in range(1, video_length-1):
                        curr_img = batch[id]['rgb']['orig'].to(self.device)
                        prev_img = batch[id-1]['rgb']['orig'].to(self.device)
                        next_img = batch[id+1]['rgb']['orig'].to(self.device)

                        orig_curr_img = denormalize(curr_img, self.device)
                        orig_imgs.append(orig_curr_img.cpu())

                        # gt_lf = batch[id]['lf'].to(self.device)
                        # gt_lfs.append(gt_lf)
                        dpt_disp = batch[id]['disp'].to(self.device)
                        disp = -1 * (dpt_disp - zp) * md
                        img = torch.cat([prev_img, curr_img, next_img, disp], dim=1)

                        img = img.to(self.device1)
                        decomposition, depth_planes, state = self.lf_model(img, prev_state)
                        pred_lf = self.val_td_model(decomposition, depth_planes)
                        pred_lf = pred_lf.clip(0, 1)
                        pred_lfs.append(pred_lf.cpu())

                        if not self.no_refinement:
                            curr_img = curr_img.to(self.device)
                            pred_lf = pred_lf.to(self.device)

                            lf_inp = torch.cat([pred_lf, curr_img.unsqueeze(1)], dim=1)
                            mask, corr_lf = self.ref_model(lf_inp)
                            ref_lf = mask*corr_lf + (1-mask)*pred_lf
                            ref_lf = ref_lf.clip(0, 1)
                            ref_lfs.append(ref_lf.cpu())

                            mask = mask.repeat(1, 1, 3, 1, 1)
                            masks.append(3*mask.cpu())
                            ref_lf_imgs = utils.lftensor2lfnp(ref_lfs)
                            for lf, path in zip(ref_lf_imgs, ref_lf_paths):
                                utils.save_video_from_lf(lf, path)
                                if self.save_numpy:
                                    np.save(path.replace('mp4', 'npy'), lf)

                        _, pred_lf_paths, ref_lf_paths, img_paths = utils.get_paths(self.save_path, i, len(pred_lfs))
                        inp_imgs = utils.imtensor2imnp(orig_imgs)
                        # gt_imgs = utils.lftensor2lfnp(gt_lfs)
                        pred_lf_imgs = utils.lftensor2lfnp(pred_lfs)
                        
                        for img, path in zip(inp_imgs, img_paths):
                            imageio.imwrite(path, np.uint8(img*255))
             
                        for lf, path in zip(pred_lf_imgs, pred_lf_paths):
                            utils.save_video_from_lf(lf, path)
                            if self.save_numpy:
                                np.save(path.replace('mp4', 'npy'), lf)
                        
                        prev_state = state
                        

    def main_worker(self):        
        ###############################################################################################
        test_loader = LFDataLoader(self.args, 'test').data
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
    
    parser.add_argument('--gpu_1', default=0, type=int, help='which gpu to use')
    parser.add_argument('--gpu_2', default=0, type=int, help='which gpu to use')
    parser.add_argument('--workers', default=1, type=int, help='number of workers for data loading')

    ######################################## Dataset parameters #######################################
    parser.add_argument('--dataset', default='Pixel4_IITM', type=str, help='Dataset to train on')

    parser.add_argument('--datapath', default='/data2/raghav/datasets/Pixel4_3DP/rectified', type=str,
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
                        default='/data2/aryan/mono-eccv/test_inputs/Pixel4_3DP_skip10',
                        type=str, help='path to the folder containing filenames to use')

    parser.add_argument('--visualization_shuffle', default=False, action='store_true', help='visualize input data')


    # parser.add_argument('-ty', '--type', type=str, default='resize', 
    #                     help='whether to train with crops or resized images')

    ############################################# I/0 parameters ######################################
    parser.add_argument('-th', '--train_height', type=int, help='train height', default=352)
    parser.add_argument('-tw', '--train_width', type=int, help='train width', default=528)
    parser.add_argument('-vh', '--val_height', type=int, help='validate height', default=352)
    parser.add_argument('-vw', '--val_width', type=int, help='validate width', default=528)
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
    parser.add_argument('--no_refinement', default=True, action='store_false')

    args = parser.parse_args()

    if args.results != '.' and not os.path.isdir(args.results):
        os.makedirs(args.results)

    tester = Tester(args)
    tester.main_worker()