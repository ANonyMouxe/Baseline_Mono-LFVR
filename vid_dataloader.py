import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from einops import rearrange
from datetime import datetime
import pandas as pd
import cv2
import glob

def preprocessing_transforms(mode):
    return transforms.Compose([ProcessAndToTensor(mode=mode)])


class LFDataLoader(object):
    def __init__(self, args, mode):
        if mode == "train":
            self.training_samples = DPDataset(
                args, mode, transform=preprocessing_transforms(mode)
            )
            self.train_sampler = None
            self.data = DataLoader(
                self.training_samples,
                args.batchsize,
                shuffle=(self.train_sampler is None),
                num_workers=args.workers,
                pin_memory=True,
                sampler=self.train_sampler,
            )

        elif mode == "eval":
            self.testing_samples = DPDataset(
                args, mode, transform=preprocessing_transforms(mode)
            )
            self.eval_sampler = None
            self.data = DataLoader(
                self.testing_samples,
                args.batchsize,
                shuffle=args.visualization_shuffle,
                num_workers=1,
                pin_memory=False,
                sampler=self.eval_sampler,
            )

        elif mode == 'test':
            self.testing_samples = DPDataset(
                args, mode, transform=preprocessing_transforms(mode)
            )
            self.eval_sampler = None
            self.data = DataLoader(
                self.testing_samples,
                args.batchsize,
                shuffle=args.visualization_shuffle,
                num_workers=1,
                pin_memory=False,
                sampler=self.eval_sampler,
            )

        else:
            print("mode should be one of 'train or eval or test'. Got {}".format(mode))


class DPDataset(Dataset):
    """
    PyTorch dataset class for the dual pixel video dataset
    """

    def __init__(self, args, mode, transform=None):
        self.datapath = args.datapath  # Path to rectified dataset
        self.unrect_datapath = args.unrect_datapath  # Path to unrectified dataset

        self.use_unimatch = args.use_unimatch
        if self.use_unimatch:
            self.unimatch_disp_path = args.unimatch_disp_path
        else:
            self.unimatch_disp_path = None

        self.mode = mode

        if mode == "eval":
            with open(os.path.join(args.filenames_file_folder, 'val_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            self.height = args.val_height
            self.width = args.val_width
        elif mode == "train":
            with open(os.path.join(args.filenames_file_folder, 'train_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            print(len(self.metadata['videos']))
            self.height = args.train_height
            self.width = args.train_width
        elif mode == 'test':
            with open(os.path.join(args.filenames_file_folder, 'test_files.json'), "r") as f:
                self.metadata = json.loads(f.read())
            print(len(self.metadata['videos']))
            self.height = args.val_height
            self.width  = args.val_width
        else:
            print(f"[!] Incorrect mode: {mode} passed to DPDataset")

        for video in self.metadata["videos"]:
            if len(video["frames"]) != self.metadata["video_length"]:
                raise Exception(
                    f"All videos need to be of the length specified in the metadata. Given video {video['video_name']} is of length {len(video['frames'])}, while expected is {self.metadata['video_length']} "
                )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata["videos"])

    def __getitem__(self, index):
        video = self.metadata["videos"][index]
        # times = {
        #     'initial': [],
        #     'rgb': [],
        #     'dp': [],
        #     'dpt': [],
        # }        
        # TODO: Modify to take in account multiple views (A,B,C). Current implementation just loads B view
        video_data = []

        for frame in video["frames"]:
            frame_data = {}
            # times['initial'].append(datetime.now().timestamp())
            orig_rgb_frame = Image.open(
                os.path.join(
                    self.datapath,
                    "B",
                    self.metadata["rgb_path"],
                    video["name"],
                    frame["rgb"]["B"],
                )
            )
            
            
            orig_rgb_frame = orig_rgb_frame.resize((self.width, self.height))
            orig_rgb_frame = np.asarray(orig_rgb_frame, dtype=np.float32) / 255.0

            # times['rgb'].append(datetime.now().timestamp())
            
            left_dp = Image.open(
                os.path.join(
                    self.unrect_datapath,
                    "B",
                    self.metadata["dp_path"],
                    video["name"],
                    frame["left_dp"]["B"],
                )
            )
            left_dp = left_dp.resize((self.width, self.height))
            left_dp = np.asarray(left_dp, dtype=np.float32)[:,:,0:1] / 255.0 # Extract only one channel as image is grayscale

            right_dp = Image.open(
                os.path.join(
                    self.unrect_datapath,
                    "B",
                    self.metadata["dp_path"],
                    video["name"],
                    frame["right_dp"]["B"],
                )
            )
            right_dp = right_dp.resize((self.width, self.height))
            right_dp = np.asarray(right_dp, dtype=np.float32) [:,:,0:1]/ 255.0
            
            # times['dp'].append(datetime.now().timestamp())

            dpt_file = frame["dptdepth"]["B"]
            if self.use_unimatch:
                # get video name
                # print("[@151][vid_dataloader]: ", os.path.join(self.unimatch_disp_path, video["name"], dpt_file.split('.')[0] + '_disp.png'))    
                disp = cv2.imread(os.path.join(self.unimatch_disp_path, video["name"], dpt_file.split('.')[0] + '_disp.png'), 
                                  cv2.IMREAD_ANYDEPTH)
                disp = cv2.resize(disp, (self.width, self.height), interpolation=cv2.INTER_CUBIC) / 255.
            
            else:
                disp = np.array(Image.open(os.path.join(
                    self.datapath, "B", "dptdepth", dpt_file)).resize((self.width, self.height)), dtype=np.float32) / 255.

            disp = disp[..., None] # Add channel dimension

            # times['dpt'].append(datetime.now().timestamp())
            
            frame_data["rgb"] = {"orig": orig_rgb_frame, "transformed": orig_rgb_frame}
            frame_data["left_dp"] = left_dp
            frame_data["right_dp"] = right_dp
            frame_data["disp"] = disp

            if self.transform:
                frame_data = self.transform(frame_data)

            video_data.append(frame_data)

        # times = pd.DataFrame(times)
        # times['dpt'] = times['dpt'] - times['dp']
        # times['dp'] = times['dp'] - times['rgb']
        # times['rgb'] = times['rgb'] - times['initial']
        # print(times)

        return video_data


class ProcessAndToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize_rgb = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.normalize_dp = transforms.Normalize(mean=[0.456], std=[0.224])

    def __call__(self, frame):
        rgb = frame["rgb"]["transformed"]
        rgb = self.to_tensor(rgb)
        rgb = self.normalize_rgb(rgb)
                
        
        orig_rgb = frame["rgb"]["orig"]
        orig_rgb = self.to_tensor(orig_rgb)

        left_dp = frame["left_dp"]
        left_dp = self.to_tensor(left_dp)
        left_dp = self.normalize_dp(left_dp)

        right_dp = frame["right_dp"]
        right_dp = self.to_tensor(right_dp)
        right_dp = self.normalize_dp(right_dp)

        disp = frame["disp"]
        disp = self.to_tensor(disp)
        
        rgb_with_dp = torch.cat([rgb, left_dp, right_dp], dim=0)

        frame_data = {
            "rgb": {"orig": orig_rgb},
            "rgb_with_dp": rgb_with_dp,
            "disp": disp,
        }
        return frame_data

    def to_tensor(self, pic):
        image = torch.FloatTensor(pic)
        shape = image.shape
        
        if len(shape) == 3:
            image = rearrange(image, "h w c -> c h w")  
            return image


if __name__ == "__main__":

    class Args:
        def __init__(self):
            # self.filenames_file = "train_inputs/train_skip5.json" # OLD STYLE, will fail
            self.filenames_file_folder = "./train_inputs/Pixel4_3DP_skip10"
            self.datapath = "/data2/raghav/datasets/Pixel4_3DP/rectified"
            self.unrect_datapath = "/data2/raghav/datasets/Pixel4_3DP/unrectified"
            self.train_height = 600
            self.train_width = 800

    dataset = DPDataset(Args(), "train", transform=preprocessing_transforms("train"))
    dLoader = DataLoader(dataset=dataset, batch_size=2)
    print(len(dataset))
    print(len(dLoader))
    
    for i, data in enumerate(dLoader):
        print(data[7]["disp"].shape)
        break