import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF 
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline_img = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        self.tf_pipeline_modal = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

        self.lidar_root = '/data/lrd/dataset/kitti360/KITTI-360'
        self.voxel_size = np.array([0.05, 0.05, 0.05])
        self.pc_range = np.array([-100, -100, -20, 100, 100, 20]) # [min_x, min_y, min_z, max_x, max_y, max_z]
        self.calib_dir = os.path.join(self.lidar_root, 'calibration')
        self.init_calibration()

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = seg_image.to(torch.uint8)
        pil_image = Image.fromarray(image.numpy())
        return pil_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
    
    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img
    

    def parse_matrix(self, line, shape):
        """解析标定文件中的矩阵行"""
        content = line.split(':')[1].strip()
        nums = [float(x) for x in content.split()]
        return np.array(nums).reshape(shape)
    
    def init_calibration(self):
        """加载标定参数并计算最终投影矩阵 Proj_Mat"""
        # 1. 加载外参 Tr_cam2velo 并求逆得到 Tr_velo2cam
        cam2velo_path = os.path.join(self.calib_dir, 'calib_cam_to_velo.txt')
        with open(cam2velo_path, 'r') as f:
            content = f.read().replace('\n', ' ')
            # 使用更健壮的数字提取
            import re
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            nums = [float(x) for x in nums]
            Tr_cam2velo = np.array(nums[:12]).reshape(3, 4)
        
        Tr_cam2velo = np.vstack([Tr_cam2velo, [0, 0, 0, 1]])
        self.Tr_velo2cam = np.linalg.inv(Tr_cam2velo)

        # 2. 加载 P_rect (投影矩阵) 和 R_rect (修正矩阵)
        intrinsic_path = os.path.join(self.calib_dir, 'perspective.txt')
        P_rect, R_rect = None, None
        with open(intrinsic_path, 'r') as f:
            for line in f:
                # 动态选择 P_rect_00 或 P_rect_01
                if line.startswith(f'P_rect_00:'):
                    P_rect_00 = self.parse_matrix(line, (3, 4))
                elif line.startswith(f'P_rect_01'):
                    P_rect_01 = self.parse_matrix(line, (3, 4))
                # KITTI-360 中所有透视相机共用 R_rect_00
                elif line.startswith('R_rect_00:'):
                    R_rect = self.parse_matrix(line, (3, 3))
        
        # 3. 计算综合投影矩阵 Proj_Mat = P * R * T
        # 统一转换为 4x4 方便矩阵运算
        R_rect_4x4 = np.eye(4)
        R_rect_4x4[:3, :3] = R_rect
        
        # 最终矩阵：LiDAR点 -> 相机像素
        self.Proj_Mat_00 = P_rect_00 @ R_rect_4x4 @ self.Tr_velo2cam
        self.Proj_Mat_01 = P_rect_01 @ R_rect_4x4 @ self.Tr_velo2cam

    def _load_cyfeat(self, lidar_path, cam_id, H, W):
        lidar_feat = np.load(lidar_path, allow_pickle=False)
        v_feats, v_coords = torch.from_numpy(lidar_feat['voxel_feats']).to(torch.float), torch.from_numpy(lidar_feat['coors']).to(torch.float)
        pts_lidar = torch.zeros((v_coords.shape[0], 4))
        pts_lidar[:, 0] = v_coords[:, 0] * self.voxel_size[0] + self.pc_range[0] # X
        pts_lidar[:, 1] = v_coords[:, 1] * self.voxel_size[1] + self.pc_range[1] # Y
        pts_lidar[:, 2] = v_coords[:, 2] * self.voxel_size[2] + self.pc_range[2] # Z
        pts_lidar[:, 3] = 1.0
        if cam_id == 'image_00':
            pts_2d = pts_lidar @ self.Proj_Mat_00.T # [N, 3]
        elif cam_id == 'image_01':
            pts_2d = pts_lidar @ self.Proj_Mat_01.T # [N, 3]
        mask = pts_2d[:, 2] > 0 # 剔除相机背后的点
        pts_2d = pts_2d[mask]
        v_feats = v_feats[mask]
        u = (pts_2d[:, 0] / pts_2d[:, 2]).to(torch.long)
        v = (pts_2d[:, 1] / pts_2d[:, 2]).to(torch.long)
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, v_feats = u[valid_mask], v[valid_mask], v_feats[valid_mask]
        C_feat = v_feats.shape[1]
        img_lidar_feat = torch.zeros((C_feat, H, W))
        img_lidar_feat[:, v, u] = v_feats.T
        return img_lidar_feat
        pass

    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            x1 = img_fname.replace('/img', '/hha').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/lidar').replace('_rgb', '_lidar')
            x3 = img_fname.replace('/img', '/event').replace('_rgb', '_event')
            lbl_path = img_fname.replace('/img', '/semantic').replace('_rgb', '_semantic')
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            # x1 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_hha'))
            # x2 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_lidar'))
            # x2 = x2.replace('.png', '_color.png')
            # x3 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_event'))
            # x3 = x3.replace('/image_00/data_rect/', '/').replace('.png', '_event_image.png')
            # lbl_path = os.path.join(*[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])

            x_lidar = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_lidar'))
            x_lidar = x_lidar.replace('.png', '_color.png')
            lbl_path = os.path.join(*[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])
        elif cfg['DATASET']['NAME'] == 'MyKITTI360':
            # img_fname: '/data/lrd/dataset/kitti360/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000386.png'
            cam_id = img_fname.split('/')[-3]
            seq_id, frame_id = img_fname.split('/')[-4], img_fname.split('/')[-1].split('.')[0]
            lidar_path = os.path.join(self.lidar_root, f'data_3d_cyfeats/{seq_id}/{frame_id}.npz')
            lbl_path = os.path.join(*[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])

        if cfg['DATASET']['NAME'] == 'DELIVER':
            image = io.read_image(img_fname)[:3, ...]
            img = self.tf_pipeline_img(image).to(self.device)
            # --- modals
            x1 = self._open_img(x1)
            x1 = self.tf_pipeline_modal(x1).to(self.device)
            x2 = self._open_img(x2)
            x2 = self.tf_pipeline_modal(x2).to(self.device)
            x3 = self._open_img(x3)
            x3 = self.tf_pipeline_modal(x3).to(self.device)
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
            label[label==255] = 0
            label -= 1

            sample = [img, x1, x2, x3][:len(modals)]

            seg_map = self.model_forward(sample)
            seg_map = self.postprocess(image, seg_map, overlay)
            return seg_map
        
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            image = io.read_image(img_fname)[:3, ...]
            img = self.tf_pipeline_img(image).to(self.device)
            # --- modals
            x_lidar = self._open_img(x_lidar)
            x_lidar = self.tf_pipeline_modal(x_lidar).to(self.device)
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
            label[label==255] = 0
            label -= 1

            sample = [img, x_lidar][:len(modals)]

            seg_map = self.model_forward(sample)
            seg_map = self.postprocess(image, seg_map, overlay)
            return seg_map

        elif cfg['DATASET']['NAME'] == 'MyKITTI360':
            image = io.read_image(img_fname)[:3, ...]
            img = self.tf_pipeline_img(image).to(self.device)
            # print(f'img.shape; {img.shape}')
            H, W = img.shape[2], img.shape[3]
            cy_feat = self._load_cyfeat(lidar_path, cam_id, H, W)
            cy_feat = self.tf_pipeline_modal(cy_feat).to(self.device)
            label = io.read_image(lbl_path)[0,...].unsqueeze(0)
            label[label==255] = 0
            label -= 1

            sample = [img, cy_feat][:len(modals)]

            seg_map = self.model_forward(sample)
            seg_map = self.postprocess(image, seg_map, overlay)
            return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # cases = ['cloud', 'fog', 'night', 'rain', 'sun', 'motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres', None]
    cases = ['lidarjitter']

    modals = cfg['DATASET']['MODALS']

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    # print(f"Model {cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}")
    # print(f"Model {cfg['DATASET']['NAME']}")

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results' / (cfg['DATASET']['NAME']+'_'+cfg['MODEL']['BACKBONE']+'_'+modals_name)
    print(f'save_dir: {save_dir}')
    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"{str(test_file.stem)}.png")
    else:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', '*', 'val', '*', '*.png']))) # --- Deliver
        # elif cfg['DATASET']['NAME'] == 'KITTI360':
        elif cfg['DATASET']['NAME'] in ('MyKITTI360', 'KITTI360'):
            source = os.path.join(test_file, 'val.txt')
            files = []
            with open(source) as f:
                files_ = f.readlines()
            for item in files_:
                file_name = item.strip()
                if ' ' in file_name:
                    # --- KITTI-360
                    file_name = os.path.join(*[str(test_file), file_name.split(' ')[0]])
                files.append(file_name)
            files = files[::10]
            # files = files[0:2]
        else:
            raise NotImplementedError()

        for file in files:
            print(file)
            if not '2013_05_28_drive_0000_sync' in file:
                continue
            segmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            print(f'save_dir: {save_dir}, file: {file}')
            file = '/'.join(file.split('/')[-4:])
            save_path = os.path.join(str(save_dir),file)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            segmap.save(save_path)
            print(f'saved to {save_path}')
        
        # wget https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync_image_00.zip
