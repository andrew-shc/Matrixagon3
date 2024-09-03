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

        self.feat = self.sdf.repeat(1, 3, 1, 1, 1)   # R,G,B
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
            tvec=np.array(image["tvec"]),
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
        self.radiance_mlp_in = nn.Linear(in_features=3, out_features=radiance_mlp_size)
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
        x_i: a simple list of 3-D x-points in the current t_i
        """
        is_f, is_b = self.sector(x_i)

        norm_fx_i = ((x_i[is_f]-self.fg_V.center)/self.fg_V.scale).float()
        norm_fx_i = torch.flip(  # X Y Z -> Z Y X
            norm_fx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        f_sdf_i = F.grid_sample(
            self.fg_V.sdf, norm_fx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).flatten()

        norm_bx_i = ((x_i[is_b]-self.bg_V.center)/self.bg_V.scale).float()
        norm_bx_i = torch.flip(  # X Y Z -> Z Y X
            norm_bx_i[torch.newaxis, torch.newaxis, torch.newaxis],
            dims=[-1]
        )
        b_sdf_i = F.grid_sample(
            self.bg_V.sdf, norm_bx_i,
            mode="bilinear", padding_mode="border", align_corners=True
        ).flatten()

        sdf = torch.full(x_i.shape[:1], 1.0)
        sdf[is_f] = f_sdf_i
        sdf[is_b] = b_sdf_i

        return sdf


    def L_o(self, x_i: torch.Tensor, v: torch.Tensor):
        pass

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ray queried position, ray direction S^2 => rendered images
        """
        N = 50

        C_img = torch.Tensor()
        T_i = torch.Tensor()  # start with 1
        for i in range(N):
            α_i = torch.max(
                0,
                (
                    torch.sigmoid(self.S(x[:, i  ]))-
                    torch.sigmoid(self.S(x[:, i+1]))
                )/torch.sigmoid(self.S(x[:, i]))
            )
            C_img += T_i*α_i*self.L_o(x[:, i], -v)
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

    def query_points_and_direction(
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

        rot_mat = quaternion.as_rotation_matrix(pose.qvec).T   # 3x3 rot mat


        # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L715
        # https://calib.io/blogs/knowledge-base/camera-models?srsltid=AfmBOoosTUXUe3QZqSrWoJXC9Yr04axC6Mvx7ru4xjo-yHMRf4H_erhx

        # initial point grid from image information
        px, py = np.mgrid[-pose.width/2:pose.width/2:_pixel_grid_width_step,
                          -pose.height/2:pose.height/2:_pixel_grid_height_step]
        pz = np.full(px.shape, 1.0)

        ##### FOCAL LENGTH NORMALIZATION (f)
        pose.f *= _scale
        img_points = np.dstack([(px.flatten()+0.5)/pose.f,
                                (py.flatten()+0.5)/pose.f,
                                pz.flatten()])[0]  # Nx2 points to be rot'd at origin

        ##### DISTORTION (k)
        distortion = pose.k*(img_points[:,0]**2+img_points[:,1]**2)

        dx = img_points[:,0]*(1+distortion)
        dy = img_points[:,1]*(1+distortion)
        dz = img_points[:,2]
        distorted_img_points = np.dstack([dx, dy, dz])[0]

        ##### ROTATION (qvec)
        rotated_img_points = np.einsum("ij,nj->ni", rot_mat, distorted_img_points)

        rpx, rpy, rpz = np.split(rotated_img_points, 3, axis=1)
        rpx, rpy, rpz = rpx.flatten(), rpy.flatten(), rpz.flatten()

        # sampling query points
        t = np.expand_dims(1.0+t_size + np.arange(0, N_query_points)*t_size, axis=1)
        expand = lambda p: np.repeat(np.expand_dims(p, axis=0), N_query_points, axis=0)

        spx = t*expand(rpx)
        spy = t*expand(rpy)
        spz = t*expand(rpz)

        imx = (px.flatten()+pose.width/2).astype(int)
        imy = (py.flatten()+pose.height/2).astype(int)
        target_pixels = data.data[imy, imx].astype(float)/255

        # path of x_i points for one viewing direction
        np.dstack([spx[:, 0], spy[:, 0], spz[:, 0]])

        sp = np.array([spx, spy, spz])  # 3xTxN
        sto = np.transpose(sp, (2, 1, 0))  # NxTx3

        # https://en.wikipedia.org/wiki/Spherical_coordinate_system
        # theta: inclination (0-180deg), 0deg = +Y; 180deg = -Y
        # phi: azimuth (0-360deg), 0deg = +X 0Z; 90deg = 0X +Z; 180deg = -X 0Z
        sto_0 = sto[:, 0]
        theta = np.arccos(sto_0[:,1] / np.linalg.norm(sto_0, axis=1))  #*(180.0/np.pi)
        phi = np.sign(sto_0[:,2])*np.arccos(sto_0[:,0] / np.linalg.norm(sto_0[:, [0,2]], axis=1))  #*(180.0/np.pi)
        v = np.stack([theta, phi], axis=1)

        # https://github.com/colmap/colmap/blob/main/src/colmap/geometry/rigid3.h#L72
        st = sto+np.einsum("ij,j->i", rot_mat, -pose.tvec)

        return (target_pixels, st, v)
    
    @staticmethod
    def S(x: np.ndarray, v_sdf: np.ndarray):
        pass

    @staticmethod
    def L_o(x: np.ndarray, v: np.ndarray, v_feat: np.ndarray):
        """
        For sake of implementation: v must be represented in S^2 unit sphere (theta and phi)
        """
        pass

    def render_color(self, query_points: np.ndarray):
        pass

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

