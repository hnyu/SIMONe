import os.path
import pickle
import torch
import glob
from torch.utils.data import Dataset
import numpy as np
import random
import re
from PIL import Image, ImageFile
from typing import Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MoviDataset(Dataset):
    def __init__(self, data_dir, img_size, seq_len=None, train=False):

        self._img_size = img_size
        self.dataset_root = os.path.expanduser(data_dir)
        self.phase_train = train

        self.video_len = 24
        self._training_videos = glob.glob(os.path.join(self.dataset_root, "train/*.pkl"))
        self._val_videos = glob.glob(os.path.join(self.dataset_root, "validation/*.pkl"))

        print("Training videos: ", len(self._training_videos))
        print("Val videos: ", len(self._val_videos))

        if self.phase_train:
            seq_len = seq_len or self.video_len
            # For training, any segment of length 'seq_len' of a video can be used
            self.num_data = len(self._training_videos) * (self.video_len - seq_len + 1)
        else:
            # For testing, each video is a sample
            self.num_data = len(self._val_videos)

        self._seq_len = seq_len
        print("Num samples: ", self.num_data)

    def __getitem__(self, index):
        if self.phase_train:
            video_i = index // (self.video_len - self._seq_len + 1)
            frame_i = index % (self.video_len - self._seq_len + 1)
            with open(self._training_videos[video_i], 'rb') as f:
                video = pickle.load(f)
        else:
            video_i = index
            frame_i = 0
            with open(self._val_videos[video_i], 'rb') as f:
                video = pickle.load(f)

        image_list = []
        seg_list = []
        for i in range(self._seq_len if self.phase_train else self.video_len):
            im = video['video'][i + frame_i]
            im = Image.fromarray(im)
            im = im.resize((self._img_size, self._img_size), resample=Image.BILINEAR)
            im_tensor = torch.from_numpy(np.array(im).astype(np.float32) / 255).permute(2, 0, 1) # [CHW]
            image_list.append(im_tensor)

            seg = video['segmentations'][i + frame_i].squeeze(-1) # [HW]
            seg = Image.fromarray(seg)
            seg = seg.resize((self._img_size, self._img_size), resample=Image.NEAREST)
            seg_tensor = torch.from_numpy(np.array(seg)) # [HW]
            seg_list.append(seg_tensor)

        img = torch.stack(image_list, dim=0)
        segm = torch.stack(seg_list, dim=0)

        # simone model uses a precision of float16
        # img: [TCHW], segm: [THW]
        return img.to(torch.float16), segm

    def __len__(self):
        return self.num_data


class MoviDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int,
        val_batch_size: int,
        val_dataset_size: int,
        seq_len: int = None,
        num_train_workers: int = 4,
    ):
        super().__init__()
        self._data_path = data_path
        self._seq_len = seq_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_train_workers = num_train_workers
        self.val_dataset_size = val_dataset_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MoviDataset(data_dir=self._data_path, img_size=64,
                                         seq_len=self._seq_len, train=True)
        self.val_dataset = MoviDataset(data_dir=self._data_path, img_size=64,
                                       train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_train_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )
