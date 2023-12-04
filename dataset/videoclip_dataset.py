import glob
import math
import os
import pickle
import cv2
import numpy as np
import torchvision
import torch
import torchvision.transforms as tf
import itertools
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class VideoClipDataset(Dataset):
    def __init__(self, config, dataset_config, clip_paths: list, phase='train'):
        self.clip_paths = clip_paths
        self.return_audio = config.network.AV_DST
        self.dataset_config = dataset_config
        self.w = dataset_config.w
        self.h = dataset_config.h
        self.pano_shape = (self.h, self.w)
        self.augmentation = phase == 'train' and config.train.use_data_augmentation

        if self.augmentation:
            self.transform = tf.Compose([
                # tf.ColorJitter(0.4, 0.4, 0.4, 0.2),
                tf.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
        else:
            self.transform = tf.Compose([
                tf.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

        self.clips = torch.stack([torch.stack(VideoRead(clip_path)) for clip_path in clip_paths])  # N, F, 3, 960, 1920
        self.n_clips = self.clips.shape[0]
        self.clip_len = self.clips.shape[1]
        self.clip_counter = self.clip_len - dataset_config.videoclip.overlap
        self.T = dataset_config.videoclip.T

        video_ids = [os.path.basename(clip_path).split('_0')[0] for clip_path in clip_paths]
        self.clip_ids = [int(os.path.basename(clip_path).split('_0')[1].split('.')[0]) for clip_path in clip_paths]
        self.gts_path = [sorted(glob.glob(f"{self.dataset_config.path_gts}/{clip_id}" + f"/*.png"))
                         for clip_id in video_ids]
        self.gtf_path = [sorted(glob.glob(f"{self.dataset_config.path_gtf}/{clip_id}" + f"/*.png"))
                         for clip_id in video_ids]

        self.gt_fps_ctr = []
        gt_fps_ctr = pickle.load(open(dataset_config['videoclip']['gt_fps_dict'], 'rb'))  # video_id: gt_fps_ctr

        for video_id in video_ids:
            self.gt_fps_ctr.append(gt_fps_ctr[video_id])

    def __len__(self):
        return self.clip_len - self.T

    def __getitem__(self, idx):
        """
        returns: [erp_img, erp_depth], erp_gt, erp_depth, filename
        """
        st, en = idx, idx + self.T
        erp_input = self.transform(self.clips[:, st:en].float() / 255.)

        erp_gt = []
        
        for c in range(self.n_clips):
            gt_idx = math.floor(self.gt_fps_ctr[c] * (en + self.clip_ids[c] * self.clip_counter)) - 1
            gts_abs_path = self.gts_path[c][gt_idx]
            gtf_abs_path = self.gtf_path[c][gt_idx]

            erp_gts = cv2.imread(gts_abs_path, 0)
            erp_gts = (erp_gts - np.min(erp_gts)) / (np.max(erp_gts) - np.min(erp_gts) + np.finfo(np.float).eps)
            erp_gts = torch.from_numpy(erp_gts)# .contiguous().to(dtype=torch.float32)

            # fixations
            erp_gtf = cv2.imread(gtf_abs_path, 0)
            erp_gtf = torch.from_numpy(erp_gtf)#.contiguous().to(dtype=torch.float32)
            erp_gt.append(torch.stack([erp_gts, erp_gtf]))
        erp_gt = torch.stack(erp_gt)

        return erp_input, erp_gt


def denorm(s):
    return s * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])


def imshow(s, title=''):
    plt.imshow(s)
    plt.title(title)
    plt.show()


def VideoRead(video_path):
    video_object = torchvision.io.VideoReader(video_path, 'video')
    start = 0
    end = float("inf")
    return [frame['data'] for frame in itertools.takewhile(lambda x: x['pts'] <= end, video_object.seek(start))]
