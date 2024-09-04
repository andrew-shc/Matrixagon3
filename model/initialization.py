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

import util



@dataclass
class ImageData:
    data: torch.Tensor

@dataclass
class ImagePose:
    width: int
    height: int
    f: float
    k: float
    tvec: torch.Tensor
    qvec: np.quaternion

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
    grid_pos: torch.Tensor
    step_size: float = field(init=False)
    voxel_pos: torch.Tensor = field(init=False)
    voxel_size: int = field(init=False)
    center: torch.Tensor= field(init=False)
    scale: torch.Tensor = field(init=False)
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
        self.voxel_pos = torch.stack((
            self.grid_pos[0, :, 0, 0],
            self.grid_pos[1, 0, :, 0],
            self.grid_pos[2, 0, 0, :],
        ), dim=0)
        self.voxel_size = self.voxel_pos.shape[1]

        self.center =(self.voxel_pos.max(dim=1).values+self.voxel_pos.min(dim=1).values)/2.0
        self.scale = (self.voxel_pos.max(dim=1).values-self.voxel_pos.min(dim=1).values)/2.0



class TargetImageDataset(Dataset):
    def __init__(self, source_path: Path, image_pose_file: Path, scale: float):
        self.source_path = source_path
        self.scale = scale

        with open(image_pose_file , "r") as fobj:
            self.image_pose = json.load(fobj)

    def __len__(self):
        return len(self.image_pose)
    
    def __getitem__(self, index) -> tuple[ImageData, ImagePose]:
        image = self.image_pose[index]
        data = cv2.imread(self.source_path / image["file_name"], cv2.IMREAD_COLOR)
        reduced = cv2.resize(data, (0, 0), fx=self.scale, fy=self.scale)

        # print(index, image["file_name"])

        return ImageData(
            data=torch.tensor(reduced/255, requires_grad=True).float()
        ), ImagePose(
            width=image["width"]*self.scale,
            height=image["height"]*self.scale,
            f=image["f"],
            k=image["k"],
            tvec=torch.Tensor(image["tvec"]),
            qvec=np.quaternion(*image["qvec"]),
        )



@torch.no_grad
def fg_bg_categorize(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

@torch.no_grad
def grid_doubling(vx: VoxelSector, scale_factor=2) -> VoxelSector:
    newx, newy, newz = torch.meshgrid(
        torch.linspace(vx.min_x, vx.max_x, steps=vx.voxel_size*2),
        torch.linspace(vx.min_y, vx.max_y, steps=vx.voxel_size*2),
        torch.linspace(vx.min_z, vx.max_z, steps=vx.voxel_size*2),
        indexing="ij",
    )
    new_grid_pos = torch.stack((newx, newy, newz))

    new_sdf = F.interpolate(vx.sdf, scale_factor=(scale_factor,)*3,
                            mode="nearest")  # , align_corners=True
    
    return VoxelSector(new_grid_pos, new_sdf)


class S(nn.Module):
    def __init__(self, fg_V: VoxelSector, bg_V: VoxelSector, device: torch.device):
        super().__init__()
        self.device = device

        self.fg_V = fg_V
        self.bg_V = bg_V

        self.fg_sdf = nn.Parameter(self.fg_V.sdf)
        self.bg_sdf = nn.Parameter(self.bg_V.sdf)

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        x_i: a simple list of 3-D x-points from current t_i
        returns: SDF occupancy
        """
        is_f, is_b = fg_bg_categorize(self, x_i)

        # foreground
        f_sdf_i = F.grid_sample(
            self.fg_sdf,
            torch.flip(  # X Y Z -> Z Y X
                # normalize
                (x_i[is_f]-self.fg_V.center)/self.fg_V.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).flatten()

        # background
        b_sdf_i = F.grid_sample(
            self.bg_sdf,
            torch.flip(  # X Y Z -> Z Y X
                # normalize
                (x_i[is_b]-self.bg_V.center)/self.bg_V.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).flatten()

        sdf = torch.full((x_i.shape[0],), 1.0).to(self.device)
        sdf[is_f] = f_sdf_i
        sdf[is_b] = b_sdf_i

        return sdf
    


class L_o(nn.Module):
    def __init__(self, fg_V: VoxelSector, bg_V: VoxelSector, radiance_mlp_size: int,
                  device: torch.device):
        super().__init__()
        self.device = device

        self.fg_V = fg_V
        self.bg_V = bg_V

        self.fg_feat = nn.Parameter(self.fg_V.feat)
        self.bg_feat = nn.Parameter(self.bg_V.feat)
        
        # R,G,B,theta,phi => R,G,B
        self.radiance_mlp_in = nn.Linear(in_features=5, out_features=radiance_mlp_size).to(self.device)
        self.radiance_mlp_out = nn.Linear(in_features=radiance_mlp_size, out_features=3).to(self.device)


    def forward(self, x_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        x_i: ''
        v: viewing direction in S^2 (theta, phi) for the corresponding x_i
        returns: predicted RGB
        """
        
        # -v
        v[0] = -v[0]
        v[1] = torch.pi-v[1]

        is_f, is_b = fg_bg_categorize(self, x_i)

        # foreground
        f_feat_i = F.grid_sample(
            self.fg_feat,
            torch.flip(  # X Y Z -> Z Y X
                (x_i[is_f]-self.fg_V.center)/self.fg_V.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze()  # CxN
        f_embedding = torch.cat([f_feat_i, v[:, is_f]], axis=0)

        # background
        b_feat_i = F.grid_sample(
            self.bg_feat,
            torch.flip(  # X Y Z -> Z Y X
                (x_i[is_b]-self.bg_V.center)/self.bg_V.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze()  # CxN
        b_embedding = torch.cat([b_feat_i, v[:, is_b]], axis=0)

        # d for default
        d_feat_i = torch.full((3,x_i.shape[0]), 0.5).to(self.device)
        embedding = torch.cat([d_feat_i, v], axis=0)
        embedding[:, is_f] = f_embedding
        embedding[:, is_b] = b_embedding

        radiance = F.relu(self.radiance_mlp_in(embedding.T))
        radiance = self.radiance_mlp_out(radiance)
        
        return radiance



class NeuralSurfaceReconstructor(nn.Module):
    """ (I)
    Parameters:
    - Foreground/background voxel SDFs
    - Foreground/background voxel feats
    - 1 hidden layer MLP w/ ReLU
    """

    def __init__(self, fg_V: VoxelSector, bg_V: VoxelSector, radiance_mlp_size: int,
                 device: torch.device):
        super().__init__()
        self.device = device

        self.fg_V = fg_V
        self.bg_V = bg_V

        # self.fg_sdf = nn.Parameter(self.fg_V.sdf)
        # self.fg_feat = nn.Parameter(self.fg_V.feat)
        # self.bg_sdf = nn.Parameter(self.bg_V.sdf)
        # self.bg_feat = nn.Parameter(self.bg_V.feat)

        self.s = S(fg_V, bg_V, device)
        self.lo = L_o(fg_V, bg_V, radiance_mlp_size, device)

        # # R,G,B,theta,phi => R,G,B
        # self.radiance_mlp_in = nn.Linear(in_features=5, out_features=radiance_mlp_size).to(self.device)
        # self.radiance_mlp_out = nn.Linear(in_features=radiance_mlp_size, out_features=3).to(self.device)

    # @torch.no_grad
    # def sector(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     returns: (is foreground, is background) both are mutually exclusive
    #     """

    #     ft = ((self.fg_V.min_x < x_i[:, 0]) & (x_i[:, 0] < self.fg_V.max_x) &
    #           (self.fg_V.min_y < x_i[:, 1]) & (x_i[:, 1] < self.fg_V.max_y) &
    #           (self.fg_V.min_z < x_i[:, 2]) & (x_i[:, 2] < self.fg_V.max_z))
    #     # background filter should have a larger bbox
    #     bt = ((self.bg_V.min_x < x_i[:, 0]) & (x_i[:, 0] < self.bg_V.max_x) &
    #           (self.bg_V.min_y < x_i[:, 1]) & (x_i[:, 1] < self.bg_V.max_y) &
    #           (self.bg_V.min_z < x_i[:, 2]) & (x_i[:, 2] < self.bg_V.max_z))

    #     return (ft, bt & ~ft)


    # def S(self, x_i: torch.Tensor) -> torch.Tensor:
    #     """
    #     x_i: a simple list of 3-D x-points from current t_i
    #     returns: SDF occupancy
    #     """
    #     is_f, is_b = self.sector(x_i)

    #     # foreground
    #     f_sdf_i = F.grid_sample(
    #         self.fg_sdf,
    #         torch.flip(  # X Y Z -> Z Y X
    #             # normalize
    #             (x_i-self.fg_V.center)/self.fg_V.scale,
    #             dims=[-1]
    #         )[torch.newaxis, torch.newaxis, torch.newaxis],
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True
    #     ).flatten()

    #     # background
    #     b_sdf_i = F.grid_sample(
    #         self.bg_sdf,
    #         torch.flip(  # X Y Z -> Z Y X
    #             # normalize
    #             (x_i-self.bg_V.center)/self.bg_V.scale,
    #             dims=[-1]
    #         )[torch.newaxis, torch.newaxis, torch.newaxis],
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True
    #     ).flatten()

    #     sdf = torch.full((x_i.shape[0],), 1.0).to(self.device)
    #     sdf = f_sdf_i
    #     sdf = b_sdf_i

    #     return sdf


    # def L_o(self, x_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    #     """
    #     x_i: ''
    #     v: viewing direction in S^2 (theta, phi) for the corresponding x_i
    #     returns: predicted RGB
    #     """
        
    #     # -v
    #     v[0] *= -1
    #     v[1] = torch.pi-v[1]

    #     is_f, is_b = self.sector(x_i)

    #     # foreground
    #     f_feat_i = F.grid_sample(
    #         self.fg_feat,
    #         torch.flip(  # X Y Z -> Z Y X
    #             (x_i-self.fg_V.center)/self.fg_V.scale,
    #             dims=[-1]
    #         )[torch.newaxis, torch.newaxis, torch.newaxis],
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True
    #     ).squeeze()  # CxN
    #     f_embedding = torch.cat([f_feat_i, v], axis=0)

    #     # background
    #     b_feat_i = F.grid_sample(
    #         self.bg_feat,
    #         torch.flip(  # X Y Z -> Z Y X
    #             (x_i-self.bg_V.center)/self.bg_V.scale,
    #             dims=[-1]
    #         )[torch.newaxis, torch.newaxis, torch.newaxis],
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True
    #     ).squeeze()  # CxN
    #     b_embedding = torch.cat([b_feat_i, v], axis=0)

    #     # d for default
    #     d_feat_i = torch.full((3,x_i.shape[0]), 0.5).to(self.device)
    #     embedding = torch.cat([d_feat_i, v], axis=0)
    #     embedding = f_embedding
    #     embedding= b_embedding

    #     radiance = F.relu(self.radiance_mlp_in(embedding.T))
    #     radiance = self.radiance_mlp_out(radiance)
        
    #     return radiance


    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ray queried position, ray direction S^2 => rendered image
        """
        T = x.shape[1]  # N for number of points to traverse

        C_img = torch.full((x.shape[0],3), 0.0).to(self.device)
        T_i = torch.full((x.shape[0],1), 1.0).to(self.device)
        for i in range(T-1):
            # print(i)
            # print("C_img", util.display_mem(C_img))
            # print("T_i", util.display_mem(T_i))
            # print(torch.cuda.memory_allocated()/1000**3)
            # print(torch.cuda.memory_reserved()/1000**3)

            S_i =  self.s(x[:, i  ])
            S_i1 = self.s(x[:, i+1])

            α_i = ((F.sigmoid(S_i)-F.sigmoid(S_i1))/F.sigmoid(S_i))[:, torch.newaxis]
            α_i[α_i < 0.0] = 0.0

            C_img = C_img + T_i*α_i*self.lo(x[:, i], v)
            T_i = T_i * (1-α_i)

        return C_img



class NeuralSurfaceReconstruction:
    def __init__(self, *, data: Path, device: torch.device):
        self.device = device

        self.images = TargetImageDataset(data / "raw_images",
                                         data / "preprocessed/image_pose.json",
                                         scale=1/8)
        self.fg_V = VoxelSector(
            grid_pos=torch.from_numpy(np.load(
                data / "preprocessed/initial_v_sdf_fg.npy"
            )).float().to(self.device),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_fg.pt", weights_only=True).to(self.device),
        )
        self.bg_V = VoxelSector(
            grid_pos=torch.from_numpy(np.load(
                data / "preprocessed/initial_v_sdf_bg.npy"
            )).float().to(self.device),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_bg.pt", weights_only=True).to(self.device),
        )

        self.model = NeuralSurfaceReconstructor(self.fg_V, self.bg_V, 64, device=self.device)

    def query_data(
            self, index: int, *, N_query_points: int, t_size: float,
            _pixel_grid_width_step = 1.0, _pixel_grid_height_step = 1.0, _scale = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
        1. Nx3 array of target image's pixels (normalized RGB)
        2. NxTx3 array of the pixels' corresponding sample query points in coordinates
        3. Nx2 array of each pixel's ray direction in S^2 unit sphere
        """
        data, pose = self.images[index]

        # rot_mat = quaternion.as_rotation_matrix(pose.qvec).T.astype(np.float32)   # 3x3 rot mat
        rot_mat = torch.from_numpy(quaternion.as_rotation_matrix(pose.qvec).T.astype(np.float32)).to(self.device)

        # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L715
        # https://calib.io/blogs/knowledge-base/camera-models?srsltid=AfmBOoosTUXUe3QZqSrWoJXC9Yr04axC6Mvx7ru4xjo-yHMRf4H_erhx

        # initial point grid from image information
        x = torch.arange(-pose.width/2, pose.width/2, _pixel_grid_width_step).to(self.device)
        y = torch.arange(-pose.height/2, pose.height/2, _pixel_grid_height_step).to(self.device)
        z = torch.Tensor([1.0]).to(self.device)
        p = torch.cartesian_prod(x, y, z)

        ##### FOCAL LENGTH NORMALIZATION (f) & alignment
        pose.f *= _scale
        p[:, 0] = (p[:, 0]+0.5)/pose.f
        p[:, 1] = (p[:, 1]+0.5)/pose.f

        del x,y,z
        gc.collect()

        ##### DISTORTION (k)
        distortion = 1 + pose.k*(p[:, 0]**2+p[:, 1]**2)
        p[:, 0] *= distortion
        p[:, 1] *= distortion

        del distortion
        gc.collect()

        ##### ROTATION (qvec) & sampling query points
        t = 1.0+t_size + torch.arange(0, N_query_points, dtype=torch.float32).to(self.device)*t_size
        # expand = lambda p: np.repeat(p[:, torch.newaxis], N_query_points, axis=1)
        expand = lambda a: a[:, torch.newaxis].repeat(1, N_query_points, 1)


        sto = t[torch.newaxis, :, torch.newaxis]*expand(
            # rotation
            torch.einsum("ij,nj->ni", rot_mat, p)
        )  # 1xTx1 , NxTx3  =>  NxTx3

        del t, p  #, distorted_img_points
        gc.collect()

        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        # theta: inclination (0-180deg), 0deg = +Y; 180deg = -Y
        # phi: azimuth (0-360deg), 0deg = +X 0Z; 90deg = 0X +Z; 180deg = -X 0Z
        
        v = torch.stack([
            # THETA
            torch.acos(sto[:,0,1] / torch.linalg.norm(sto[:,0], dim=1)),  #*(180.0/np.pi)
            # PHI
            torch.sign(sto[:,0,2])*torch.acos(sto[:,0,0] / torch.linalg.norm(sto[:,0,[0,2]], dim=1)),   #*(180.0/np.pi)
        ], dim=0)

        # https://github.com/colmap/colmap/blob/main/src/colmap/geometry/rigid3.h#L72
        st = sto+torch.einsum("ij,j->i", rot_mat, -pose.tvec.to(self.device))

        return (data.data.permute(1,0,2).flatten(end_dim=1), st, v)

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

