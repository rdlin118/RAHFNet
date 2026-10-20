import os
import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import glob
import einops
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from semseg.augmentations_mm import get_train_augmentation
from semseg.datasets.kitti360 import KITTI360

import torch.multiprocessing as mp

class MyKITTI360(KITTI360):
    """
    num_classes: 19
    """
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    ID2TRAINID = {0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 
    20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 34:2, 35:4, 36:255, 37:5, 38:255, 39:255, 40:255, 41:255, 42:255, 43:255, 44:255, -1:255}

    def __init__(self, root: str = '/path/to/KITTI-360', split: str = 'train', transform = None, modals = ['img', 'lidar'], case = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.root = root
        self.lidar_root = root
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.modals = modals

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        self.files = self._get_file_names(split)
        print(f"Found {len(self.files)} {split} images.")

        self.voxel_size = np.array([0.05, 0.05, 0.05])
        self.pc_range = np.array([-100, -100, -20, 100, 100, 20])
        
        self.calib_dir = os.path.join(self.root, 'calibration')

        self.init_calibration()


    def parse_matrix(self, line, shape):
        content = line.split(':')[1].strip()
        nums = [float(x) for x in content.split()]
        return np.array(nums).reshape(shape)

    
    def init_calibration(self):
        cam2velo_path = os.path.join(self.calib_dir, 'calib_cam_to_velo.txt')
        with open(cam2velo_path, 'r') as f:
            content = f.read().replace('\n', ' ')
            import re
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            nums = [float(x) for x in nums]
            Tr_cam2velo = np.array(nums[:12]).reshape(3, 4)
        
        Tr_cam2velo = np.vstack([Tr_cam2velo, [0, 0, 0, 1]])
        self.Tr_velo2cam = np.linalg.inv(Tr_cam2velo)

        intrinsic_path = os.path.join(self.calib_dir, 'perspective.txt')
        P_rect, R_rect = None, None
        with open(intrinsic_path, 'r') as f:
            for line in f:
                if line.startswith(f'P_rect_00:'):
                    P_rect_00 = self.parse_matrix(line, (3, 4))
                elif line.startswith(f'P_rect_01'):
                    P_rect_01 = self.parse_matrix(line, (3, 4))
                elif line.startswith('R_rect_00:'):
                    R_rect = self.parse_matrix(line, (3, 3))
        
        R_rect_4x4 = np.eye(4)
        R_rect_4x4[:3, :3] = R_rect
        
        self.Proj_Mat_00 = P_rect_00 @ R_rect_4x4 @ self.Tr_velo2cam
        self.Proj_Mat_01 = P_rect_01 @ R_rect_4x4 @ self.Tr_velo2cam
    

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        item_name = str(self.files[index])
        cam_id = item_name.split('/')[-3]
        rgb = os.path.join(self.root, item_name)
        seq_id, frame_id = item_name.split('/')[1], item_name.split('/')[-1].split('.')[0]
        lidar_path = os.path.join(self.lidar_root, f'data_3d_cyfeats/{seq_id}/{frame_id}.npz')
        lidar_feat = np.load(lidar_path, allow_pickle=False)

        sample = {}
        img = io.read_image(rgb)[:3, ...]
        sample['img'] = img
        H, W = img.shape[1], img.shape[2]
        v_feats, v_coords = torch.from_numpy(lidar_feat['voxel_feats']).to(torch.float), torch.from_numpy(lidar_feat['coors']).to(torch.float)

        pts_lidar = torch.zeros((v_coords.shape[0], 4))
        pts_lidar[:, 0] = v_coords[:, 0] * self.voxel_size[0] + self.pc_range[0]
        pts_lidar[:, 1] = v_coords[:, 1] * self.voxel_size[1] + self.pc_range[1]
        pts_lidar[:, 2] = v_coords[:, 2] * self.voxel_size[2] + self.pc_range[2]
        pts_lidar[:, 3] = 1.0

        if cam_id == 'image_00':
            pts_2d = pts_lidar @ self.Proj_Mat_00.T
        elif cam_id == 'image_01':
            pts_2d = pts_lidar @ self.Proj_Mat_01.T
        
        mask = pts_2d[:, 2] > 0
        pts_2d = pts_2d[mask]
        v_feats = v_feats[mask]
        
        u = (pts_2d[:, 0] / pts_2d[:, 2]).to(torch.long)
        v = (pts_2d[:, 1] / pts_2d[:, 2]).to(torch.long)

        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, v_feats = u[valid_mask], v[valid_mask], v_feats[valid_mask]

        C_feat = v_feats.shape[1]
        img_lidar_feat = torch.zeros((C_feat, H, W))
        
        img_lidar_feat[:, v, u] = v_feats.T


        sample = {
            'img': img,
            'voxel_feat': img_lidar_feat,
        }
        lbl_path = os.path.join(*[self.root, item_name.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])
        label = io.read_image(lbl_path)[0,...].unsqueeze(0)
        sample['mask'] = label
        
        if self.transform:
            sample = self.transform(sample)
        label = sample['mask']
        del sample['mask']
        label = self.encode(label.squeeze().numpy()).long()

        return [sample['img'], sample['voxel_feat']], label
