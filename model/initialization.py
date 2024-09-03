from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import os
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import cv2
import quaternion

import gc




@dataclass
class ImageData:
    data: np.ndarray

@dataclass
class ImagePose:
    width: int
    height: int
    f: float
    k: float
    tvec: float
    qvec: float

@dataclass
class VoxelSector:  # background / foreground
    # bounding box
    min_x: float = field(init=False)
    max_x: float = field(init=False)
    min_y: float = field(init=False)
    max_y: float = field(init=False)
    min_z: float = field(init=False)
    max_z: float = field(init=False)
    # grid positioning
    grid_pos: np.ndarray
    step_size: float = field(init=False)
    voxel_pos: np.ndarray = field(init=False)
    center: np.ndarray = field(init=False)
    scale: np.ndarray = field(init=False)
    # grid data
    sdf: torch.Tensor
    feat: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.min_x = self.grid_pos[0].min()
        self.max_x = self.grid_pos[0].max()
        self.min_y = self.grid_pos[1].min()
        self.max_y = self.grid_pos[1].max()
        self.min_z = self.grid_pos[2].min()
        self.max_z = self.grid_pos[2].max()

        self.feat = self.sdf.repeat(1, 3, 1, 1, 1)   # B,C(R,G,B),X,Y,Z
        self.feat[:] = 0.5

        self.step_size = self.grid_pos[0][0][0][0]-self.grid_pos[0][1][0][0]        
        self.voxel_pos = np.array([
            self.grid_pos[0, :, 0, 0],
            self.grid_pos[1, 0, :, 0],
            self.grid_pos[2, 0, 0, :],
        ], dtype=np.float32)

        self.center = ((self.voxel_pos.max(axis=1)+self.voxel_pos.min(axis=1))/2)
        self.scale = (self.voxel_pos.max(axis=1)-self.voxel_pos.min(axis=1))/2



class TargetImageDataset(Dataset):
    def __init__(self, source_path: Path, image_pose_file: Path):
        self.source_path = source_path

        with open(image_pose_file , "r") as fobj:
            self.image_pose = json.load(fobj)

    def __len__(self):
        return len(self.image_pose)
    
    def __getitem__(self, index) -> tuple[ImageData, ImagePose]:
        image = self.image_pose[index]
        data = cv2.imread(self.source_path / image["file_name"], cv2.IMREAD_COLOR)

        # print(index, image["file_name"])

        return ImageData(
            data=data
        ), ImagePose(
            width=image["width"],
            height=image["height"],
            f=image["f"],
            k=image["k"],
            tvec=np.array(image["tvec"]).astype(np.float32),
            qvec=np.quaternion(*image["qvec"]),
        )
    

class NeuralSurfaceReconstructor(nn.Module):
    """ (I)
    Parameters:
    - Foreground/background voxel SDFs
    - Foreground/background voxel feats
    - 1 hidden layer MLP w/ ReLU
    """

    def __init__(self, fg_V: VoxelSector, bg_V: VoxelSector, radiance_mlp_size: int):
        super().__init__()

        self.fg_V = fg_V
        self.bg_V = bg_V

        self.fg_sdf = nn.Parameter(self.fg_V.sdf)
        self.fg_feat = nn.Parameter(self.fg_V.feat)
        self.bg_sdf = nn.Parameter(self.bg_V.sdf)
        self.bg_feat = nn.Parameter(self.bg_V.feat)

        # R,G,B,theta,phi => R,G,B
        self.radiance_mlp_in = nn.Linear(in_features=5, out_features=radiance_mlp_size)
        self.radiance_mlp_out = nn.Linear(in_features=radiance_mlp_size, out_features=3)

    @torch.no_grad
    def sector(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        returns: (is foreground, is background) both are mutually exclusive
        """

        ft = ((self.fg_V.min_x < x_i[:, 0]) & (x_i[:, 0] < self.fg_V.max_x) &
              (self.fg_V.min_y < x_i[:, 1]) & (x_i[:, 1] < self.fg_V.max_y) &
              (self.fg_V.min_z < x_i[:, 2]) & (x_i[:, 2] < self.fg_V.max_z))
        # background filter should have a larger bbox
        bt = ((self.bg_V.min_x < x_i[:, 0]) & (x_i[:, 0] < self.bg_V.max_x) &
              (self.bg_V.min_y < x_i[:, 1]) & (x_i[:, 1] < self.bg_V.max_y) &
              (self.bg_V.min_z < x_i[:, 2]) & (x_i[:, 2] < self.bg_V.max_z))

        return (ft, bt & ~ft)


    def S(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        x_i: a simple list of 3-D x-points from current t_i
        returns: SDF occupancy
        """
        is_f, is_b = self.sector(x_i)

        # foreground
        norm_fx_i = ((x_i[is_f]-self.fg_V.center)/self.fg_V.scale).float()
        norm_fx_i = torch.flip(  # X Y Z -> Z Y X
            norm_fx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        f_sdf_i = F.grid_sample(
            self.fg_V.sdf, norm_fx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).flatten()

        # background
        norm_bx_i = ((x_i[is_b]-self.bg_V.center)/self.bg_V.scale).float()
        norm_bx_i = torch.flip(  # X Y Z -> Z Y X
            norm_bx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        b_sdf_i = F.grid_sample(
            self.bg_V.sdf, norm_bx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).flatten()

        sdf = torch.full((x_i.shape[0],), 1.0)
        sdf[is_f] = f_sdf_i
        sdf[is_b] = b_sdf_i

        return sdf


    def L_o(self, x_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        x_i: ''
        v: viewing direction in S^2 (theta, phi) for the corresponding x_i
        returns: predicted RGB
        """
        
        # -v
        v[0] *= -1
        v[1] = torch.pi-v[1]

        is_f, is_b = self.sector(x_i)

        # foreground
        norm_fx_i = ((x_i[is_f]-self.fg_V.center)/self.fg_V.scale).float()
        norm_fx_i = torch.flip(  # X Y Z -> Z Y X
            norm_fx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        f_feat_i = F.grid_sample(
            self.fg_V.feat, norm_fx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze()  # CxN
        f_embedding = torch.cat([f_feat_i, v[:, is_f]], axis=0)

        # background
        norm_bx_i = ((x_i[is_b]-self.bg_V.center)/self.bg_V.scale).float()
        norm_bx_i = torch.flip(  # X Y Z -> Z Y X
            norm_bx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        b_feat_i = F.grid_sample(
            self.bg_V.feat, norm_bx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze()  # CxN
        b_embedding = torch.cat([b_feat_i, v[:, is_b]], axis=0)

        # d for default
        d_feat_i = torch.full((3,x_i.shape[0]), 0.5)
        embedding = torch.cat([d_feat_i, v], axis=0)
        embedding[:, is_f] = f_embedding
        embedding[:, is_b] = b_embedding

        radiance = F.relu(self.radiance_mlp_in(embedding.T))
        radiance = self.radiance_mlp_out(radiance)
        
        return radiance


    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ray queried position, ray direction S^2 => rendered image
        """
        N = 50

        C_img = torch.full((x.shape[0],), 0.0)
        T_i = torch.full((x.shape[0],), 1.0)
        for i in range(N-1):
            α_i = torch.max(
                0,
                (
                    F.sigmoid(self.S(x[:, i  ]))-
                    F.sigmoid(self.S(x[:, i+1]))
                )/F.sigmoid(self.S(x[:, i]))
            )
            C_img += T_i*α_i*self.L_o(x[:, i], v)
            T_i *= (1-α_i)



class NeuralSurfaceReconstruction:
    def __init__(self, *, data: Path):
        self.images = TargetImageDataset(data / "raw_images",
                                         data / "preprocessed/image_pose.json")
        self.fg_V = VoxelSector(
            grid_pos=np.load(data / "preprocessed/initial_v_sdf_fg.npy"),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_fg.pt", weights_only=True),
        )
        self.bg_V = VoxelSector(
            grid_pos=np.load(data / "preprocessed/initial_v_sdf_bg.npy"),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_bg.pt", weights_only=True),
        )

        self.model = NeuralSurfaceReconstructor(self.fg_V, self.bg_V, 64)

    def query_data(
            self, index: int, *, N_query_points: int, t_size: float,
            _pixel_grid_width_step = 1.0, _pixel_grid_height_step = 1.0, _scale = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        returns:
        1. Nx3 array of target image's pixels (normalized RGB)
        2. NxTx3 array of the pixels' corresponding sample query points in coordinates
        3. Nx2 array of each pixel's ray direction in S^2 unit sphere
        """
        data, pose = self.images[index]

        target_pixels = data.data.astype(np.float32)/255

        del data.data
        del data
        gc.collect()

        rot_mat = quaternion.as_rotation_matrix(pose.qvec).T.astype(np.float32)   # 3x3 rot mat


        # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L715
        # https://calib.io/blogs/knowledge-base/camera-models?srsltid=AfmBOoosTUXUe3QZqSrWoJXC9Yr04axC6Mvx7ru4xjo-yHMRf4H_erhx

        # initial point grid from image information
        px, py = np.mgrid[-pose.width/2:pose.width/2:_pixel_grid_width_step,
                          -pose.height/2:pose.height/2:_pixel_grid_height_step]
        px, py = np.float32(px), np.float32(py)

        ##### FOCAL LENGTH NORMALIZATION (f)
        pose.f *= _scale
        nx = (px.flatten()+0.5)/pose.f
        ny = (py.flatten()+0.5)/pose.f

        del px, py
        gc.collect()

        ##### DISTORTION (k)
        distortion = pose.k*(nx**2+ny**2)

        distorted_img_points = np.dstack([
            nx*(1+distortion),
            ny*(1+distortion),
            np.full(nx.shape, 1.0, dtype=np.float32),  # z
        ])[0]

        del nx,ny,distortion
        gc.collect()

        ##### DISTORTION (k) & sampling query points
        t = 1.0+t_size + np.arange(0, N_query_points, dtype=np.float32)*t_size
        expand = lambda p: np.repeat(p[:, np.newaxis], N_query_points, axis=1)

        sto = t[np.newaxis, :, np.newaxis]*expand(
            np.einsum("ij,nj->ni", rot_mat, distorted_img_points).astype(np.float32)  # rotation
        )  # 1xTx1 , NxTx3  =>  NxTx3

        del t, distorted_img_points
        gc.collect()

        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        # theta: inclination (0-180deg), 0deg = +Y; 180deg = -Y
        # phi: azimuth (0-360deg), 0deg = +X 0Z; 90deg = 0X +Z; 180deg = -X 0Z
        sto_0 = sto[:, 0]
        
        v = np.stack([
            # THETA
            np.arccos(sto_0[:,1] / np.linalg.norm(sto_0, axis=1)),  #*(180.0/np.pi)
            # PHI
            np.sign(sto_0[:,2])*np.arccos(sto_0[:,0] / np.linalg.norm(sto_0[:, [0,2]], axis=1)),   #*(180.0/np.pi)
        ], axis=0)

        del sto_0  #, theta, phi
        gc.collect()

        # https://github.com/colmap/colmap/blob/main/src/colmap/geometry/rigid3.h#L72
        st = sto+np.einsum("ij,j->i", rot_mat, -pose.tvec)

        return (torch.from_numpy(target_pixels),
                torch.from_numpy(st),
                torch.from_numpy(v))

    def train(self):
        loss = 0


    

def initialization():
    nsr = NeuralSurfaceReconstruction(
        img=Path("./data/calculator/raw_images"),
        img_pose=Path("./data/image_pose.json"),
        fg_v_sdf=Path("./data/initial_v_sdf_fg.pt"),
        bg_v_sdf=Path("./data/initial_v_sdf_bg.pt"),
    )





if __name__ == "__main__":
    initialization()

