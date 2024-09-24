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
from typing import Optional

import gc

import util



@dataclass
class ImageData:
    data: torch.Tensor

@dataclass
class ImagePose:
    width: int
    height: int
    scale: float
    f: float
    k: float
    tvec: torch.Tensor
    rmat: torch.Tensor

@dataclass
class VoxelInfo:
    # bounding box
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    # normalization
    center: torch.Tensor
    scale: torch.Tensor

@dataclass
class VoxelSector:  # background / foreground
    info: VoxelInfo = field(init=False)

    # grid positioning
    grid_pos: torch.Tensor
    step_size: float = field(init=False)
    voxel_pos: torch.Tensor = field(init=False)
    voxel_size: int = field(init=False)
    # grid data
    sdf: torch.Tensor
    feat: Optional[torch.Tensor]

    def __post_init__(self):
        if self.feat is None:
            self.feat = self.sdf.repeat(1, 3, 1, 1, 1)   # B,C(R,G,B),X,Y,Z
            self.feat[:] = 0.5

        self.step_size = self.grid_pos[0][0][0][0]-self.grid_pos[0][1][0][0]        
        self.voxel_pos = torch.stack((
            self.grid_pos[0, :, 0, 0],
            self.grid_pos[1, 0, :, 0],
            self.grid_pos[2, 0, 0, :],
        ), dim=0)
        self.voxel_size = self.voxel_pos.shape[1]

        self.info = VoxelInfo(
            min_x = self.grid_pos[0].min(),
            max_x = self.grid_pos[0].max(),
            min_y = self.grid_pos[1].min(),
            max_y = self.grid_pos[1].max(),
            min_z = self.grid_pos[2].min(),
            max_z = self.grid_pos[2].max(),
            center = (self.voxel_pos.max(dim=1).values+self.voxel_pos.min(dim=1).values)/2.0,
            scale = (self.voxel_pos.max(dim=1).values-self.voxel_pos.min(dim=1).values)/2.0,
        )



class TargetImageDataset(Dataset):
    def __init__(self, source_path: Path, image_pose_file: Path, scale: float, device):
        self.source_path = source_path
        self.scale = scale
        self.device = device

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
            data=torch.tensor(reduced/255, requires_grad=True).to(self.device).float()
        ), ImagePose(
            width=image["width"]*self.scale,
            height=image["height"]*self.scale,
            scale=self.scale,
            f=image["f"],
            k=image["k"],
            tvec=torch.tensor(image["tvec"], device=self.device),
            rmat=torch.from_numpy(
                quaternion.as_rotation_matrix(
                    np.quaternion(*image["qvec"])
                ).T.astype(np.float32)
            ).to(self.device),
        )


@torch.no_grad
def sector_splitter(
    x_i: torch.Tensor, fg: VoxelInfo, bg: VoxelInfo
) -> tuple[torch.Tensor, torch.Tensor]:
    """ [BxN]x3 => (BxN, BxN)

    # Nx3 -> BxNx3 = [:, i] -> [:, :, i]
    returns: (is foreground, is background) both are mutually exclusive
    """
    
    ft = ((fg.min_x < x_i[:,0]) & (x_i[:,0] < fg.max_x) &
        (fg.min_y < x_i[:,1]) & (x_i[:,1] < fg.max_y) &
        (fg.min_z < x_i[:,2]) & (x_i[:,2] < fg.max_z))
    # background filter should have a larger bbox
    bt = ((bg.min_x < x_i[:,0]) & (x_i[:,0] < bg.max_x) &
        (bg.min_y < x_i[:,1]) & (x_i[:,1] < bg.max_y) &
        (bg.min_z < x_i[:,2]) & (x_i[:,2] < bg.max_z))
        
    return (ft, bt & ~ft)


@torch.no_grad
def grid_scaling(vx: VoxelSector, *, scale_factor, device) -> VoxelSector:
    newx, newy, newz = torch.meshgrid(
        torch.linspace(vx.info.min_x, vx.info.max_x, steps=vx.voxel_size*2, device=device),
        torch.linspace(vx.info.min_y, vx.info.max_y, steps=vx.voxel_size*2, device=device),
        torch.linspace(vx.info.min_z, vx.info.max_z, steps=vx.voxel_size*2, device=device),
        indexing="ij",
    )
    new_grid_pos = torch.stack((newx, newy, newz))

    new_sdf = F.interpolate(vx.sdf, scale_factor=(scale_factor,)*3,
                            mode="trilinear")  # , align_corners=True
    new_feat = F.interpolate(vx.feat, scale_factor=(scale_factor,)*3,
                            mode="trilinear")  # , align_corners=True
    
    return VoxelSector(grid_pos=new_grid_pos, sdf=new_sdf, feat=new_feat)


class S(nn.Module):
    def __init__(self,
                 fg_info: VoxelInfo, fg_sdf: torch.Tensor,
                 bg_info: VoxelInfo, bg_sdf: torch.Tensor,
                 device: torch.device):
        super().__init__()
        self.device = device

        self.update_grid(
            fg_info, fg_sdf,
            bg_info, bg_sdf,
        )

    @torch.no_grad
    def update_grid(self,
                    fg_info: VoxelInfo, fg_sdf: torch.Tensor,
                    bg_info: VoxelInfo, bg_sdf: torch.Tensor):

        self.fg = fg_info
        self.bg = bg_info
        
        self.fg_sdf = nn.Parameter(fg_sdf)
        self.bg_sdf = nn.Parameter(bg_sdf)

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        """
        x_i: a simple list of 3-D x-points from current t_i
                BxNx3
        returns: SDF occupancy
                BxNx(1)
        """

        p = x_i.flatten(end_dim=1)
        is_f, is_b = sector_splitter(p, self.fg, self.bg)

        # foreground
        f_sdf_i = F.grid_sample(
            self.fg_sdf,
            torch.flip(  # [[X Y Z]] -> [[Z Y X]]
                # normalize
                (p[is_f]-self.fg.center)/self.fg.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze((0,1,2,3))   # [BxN]

        # background
        b_sdf_i = F.grid_sample(
            self.bg_sdf,
            torch.flip(  # [[X Y Z]] -> [[Z Y X]]
                # normalize
                (p[is_b]-self.bg.center)/self.bg.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze((0,1,2,3))   # [BxN]

        sdf = torch.full((x_i.shape[0]*x_i.shape[1],), 1.0, device=self.device)
        sdf[is_f] = f_sdf_i
        sdf[is_b] = b_sdf_i

        return sdf.reshape((x_i.shape[0], x_i.shape[1]))
    


class L_o(nn.Module):
    def __init__(self,
                 fg_info: VoxelInfo, fg_feat: torch.Tensor,
                 bg_info: VoxelInfo, bg_feat: torch.Tensor,
                 radiance_mlp_size: int, device: torch.device):
        super().__init__()
        self.device = device

        self.update_grid(
            fg_info, fg_feat,
            bg_info, bg_feat,
        )
        
        # R,G,B,theta,phi => R,G,B
        self.radiance_mlp_in = nn.Linear(in_features=5, out_features=radiance_mlp_size, device=self.device)
        self.radiance_mlp_out = nn.Linear(in_features=radiance_mlp_size, out_features=3, device=self.device)

    @torch.no_grad
    def update_grid(self,
                    fg_info: VoxelInfo, fg_feat: torch.Tensor,
                    bg_info: VoxelInfo, bg_feat: torch.Tensor):

        self.fg = fg_info
        self.bg = bg_info
        
        self.fg_feat = nn.Parameter(fg_feat)
        self.bg_feat = nn.Parameter(bg_feat)

    def forward(self, x_i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        x_i: BxNx3
        v: viewing direction in S^2 (theta, phi) for the corresponding x_i
            Bx2xN -> BxNx2 -> [BxN]x2
        returns: predicted RGB radiance @ i
           BxNx3
        """
        
        # -v
        v = v.transpose(dim0=1, dim1=2).flatten(end_dim=1)
        v[0] = -v[0]
        v[1] = torch.pi-v[1]

        p = x_i.flatten(end_dim=1)
        is_f, is_b = sector_splitter(p, self.fg, self.bg)

        # foreground
        f_feat_i = F.grid_sample(
            self.fg_feat,
            torch.flip(  # [[X Y Z]] -> [[Z Y X]]
                (p[is_f]-self.fg.center)/self.fg.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze((0,2,3)).T  # Cx[BxN] -> [BxN]xC
        f_embedding = torch.cat([f_feat_i, v[is_f]], axis=1)


        # background
        b_feat_i = F.grid_sample(
            self.bg_feat,
            torch.flip(  # [[X Y Z]] -> [[Z Y X]]
                (p[is_b]-self.bg.center)/self.bg.scale,
                dims=[-1]
            )[torch.newaxis, torch.newaxis, torch.newaxis],
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        ).squeeze((0,2,3)).T  # Cx[BxN]-> [BxN]xC
        b_embedding = torch.cat([b_feat_i, v[is_b]], axis=1)

        # d for default
        d_feat_i = torch.full((x_i.shape[0]*x_i.shape[1],3), 0.5, device=self.device)
        embedding = torch.cat([d_feat_i, v], axis=1)
        embedding[is_f] = f_embedding
        embedding[is_b] = b_embedding

        radiance = F.relu(self.radiance_mlp_in(embedding))
        radiance = self.radiance_mlp_out(radiance)
        
        return radiance.reshape((x_i.shape[0], x_i.shape[1], 3))
    

class NSRLoss(nn.Module):
    """ Neural Surface Reconstruction Loss
    """
    def __init__(self,
                 fg_V: VoxelSector, bg_V: VoxelSector,
                 radiance_mlp_size: int, initial_sharpness: int,
                 w_lap: float, w_pprgb: float,
                 device: torch.device):
        super().__init__()
        self.device = device

        self.s = S(fg_V.info, fg_V.sdf,
                   bg_V.info, bg_V.sdf,
                   device)
        self.lo = L_o(fg_V.info, fg_V.feat,
                      bg_V.info, bg_V.feat,
                      radiance_mlp_size, device)
        
        self.adaptive_huber_loss = torch.nn.HuberLoss(reduction="sum")
        self.neighboring_filter = torch.tensor([[[
            [[0.,0.,0.],[0.,1.,0.],[0.,0.,0.],],
            [[0.,1.,0.],[1.,0.,1.],[0.,1.,0.],],
            [[0.,0.,0.],[0.,1.,0.],[0.,0.,0.],],
        ]]], device=self.device)

        self.sharpness = initial_sharpness

        self.w_lap = w_lap
        self.w_pprgb = w_pprgb

    def σ(self, y: torch.Tensor) -> torch.Tensor:
        return 1/(1+torch.exp(-self.sharpness*y))
    
    def laplacian_regularization(self, sdf: torch.Tensor):
        padded = F.pad(sdf, pad=(1,1,1,1,1,1), mode="replicate")
        neighbor = F.conv3d(padded, self.neighboring_filter)
        return torch.square(neighbor-6*sdf).sum()

    def forward(self, tp: torch.Tensor, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ground truth pixels, ray queried position, ray direction S^2 => scalar loss value
            BxNx3,BxNxTx3,BxNx2 -> 1
        """
        T = x.shape[2]  # number of points to sample for each ray direction

        pprgb_loss = torch.tensor(0.0, device=self.device)

        C_img = torch.full((x.shape[0],x.shape[1],3), 0.0, device=self.device)
        T_i = torch.full((x.shape[0],x.shape[1],1), 1.0, device=self.device)

        S_i =  self.s(x[:,:, 0  ])
        S_i1 = None
        for i in range(T-1):
            S_i1 = self.s(x[:,:, i+1])
            L_o = self.lo(x[:,:, i], v)

            α_i = ((self.σ(S_i)-self.σ(S_i1))/self.σ(S_i))[:,:, torch.newaxis]
            α_i[α_i < 0.0] = 0.0

            C_img = C_img + T_i*α_i*L_o
            T_i = T_i * (1-α_i)

            pprgb_loss = pprgb_loss + torch.sum(T_i*α_i*abs(L_o-tp))


            S_i = S_i1

        photo_loss = self.adaptive_huber_loss(tp, C_img)
        laplacian_loss = self.laplacian_regularization(self.s.fg_sdf) + self.laplacian_regularization(self.s.bg_sdf)

        return (photo_loss + self.w_lap*laplacian_loss + self.w_pprgb*pprgb_loss,
                C_img)


class NeuralSurfaceReconstruction:
    def __init__(self, *, data: Path, device: torch.device, image_scale: float):
        self.device = device

        self.images = TargetImageDataset(data / "raw_images",
                                         data / "preprocessed/image_pose.json",
                                         scale=image_scale,
                                         device=self.device)
        self.fg_V = VoxelSector(
            grid_pos=torch.from_numpy(np.load(
                data / "preprocessed/initial_v_sdf_fg.npy"
            )).float().to(self.device),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_fg.pt", weights_only=True).to(self.device),
            feat=None,
        )
        self.bg_V = VoxelSector(
            grid_pos=torch.from_numpy(np.load(
                data / "preprocessed/initial_v_sdf_bg.npy"
            )).float().to(self.device),
            sdf=torch.load(data / "preprocessed/initial_v_sdf_bg.pt", weights_only=True).to(self.device),
            feat=None,
        )

        self.model = NSRLoss(
            self.fg_V, self.bg_V, 
            radiance_mlp_size=128, initial_sharpness=1,
            w_lap=10e-8, w_pprgb=0.01,
            device=self.device
        )

    @torch.no_grad
    def query_data(
            self, index: int, *, N_query_points: int,
            _pixel_grid_width_step = 1.0, _pixel_grid_height_step = 1.0, _scale = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
        1. Nx3 array of target image's pixels (normalized RGB)
        2. NxTx3 array of the pixels' corresponding sample query points in coordinates
        3. Nx2 array of each pixel's ray direction in S^2 unit sphere
        """
        data, pose = self.images[index]

        # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L715
        # https://calib.io/blogs/knowledge-base/camera-models?srsltid=AfmBOoosTUXUe3QZqSrWoJXC9Yr04axC6Mvx7ru4xjo-yHMRf4H_erhx

        # initial point grid from image information
        x = torch.arange(-pose.width/2, pose.width/2, _pixel_grid_width_step, device=self.device)
        y = torch.arange(-pose.height/2, pose.height/2, _pixel_grid_height_step, device=self.device)
        z = torch.tensor([1.0], device=self.device)
        p = torch.cartesian_prod(x, y, z)

        ##### FOCAL LENGTH NORMALIZATION (f) & 0.5 alignment
        pose.f *= _scale
        p[:, 0] = (p[:, 0]+0.5)/pose.f/pose.scale
        p[:, 1] = (p[:, 1]+0.5)/pose.f/pose.scale

        del x,y,z
        gc.collect()

        ##### DISTORTION (k)
        distortion = 1 + pose.k*(p[:, 0]**2+p[:, 1]**2)
        p[:, 0] *= distortion
        p[:, 1] *= distortion

        del distortion
        gc.collect()

        ##### ROTATION (qvec) & sampling query points
        t_size = abs(self.fg_V.step_size/2)
        t = 1.0+t_size + torch.arange(0, N_query_points, dtype=torch.float32, device=self.device)*t_size
        # expand = lambda p: np.repeat(p[:, torch.newaxis], N_query_points, axis=1)
        expand = lambda a: a[:, torch.newaxis].repeat(1, N_query_points, 1)


        sto = t[torch.newaxis, :, torch.newaxis]*expand(
            # rotation
            torch.einsum("ij,nj->ni", pose.rmat, p)
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
        st = sto+torch.einsum("ij,j->i", pose.rmat, -pose.tvec)

        return (data.data.permute(1,0,2).flatten(end_dim=1),
                st,
                v)
    
    @torch.no_grad
    def double_grid(self):
        self.fg_V = grid_scaling(self.fg_V, scale_factor=2, device=self.device)
        self.bg_V = grid_scaling(self.bg_V, scale_factor=2, device=self.device)

        self.model.s.update_grid(
            self.fg_V.info, self.fg_V.sdf,
            self.bg_V.info, self.bg_V.sdf,
        )
        self.model.lo.update_grid(
            self.fg_V.info, self.fg_V.feat,
            self.bg_V.info, self.bg_V.feat,
        )

    @torch.no_grad
    def increment_sharpness(self):
        if self.model.sharpness < 300:
            self.model.sharpness += 1.0

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

