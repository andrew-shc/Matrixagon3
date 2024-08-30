import torch
from torch.utils.data import Dataset
import json
import os
from dataclasses import dataclass
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

        return ImageData(
            data=data
        ), ImagePose(
            width=image["width"],
            height=image["height"],
            f=image["f"],
            k=image["k"],
            tvec=image["tvec"],
            qvec=np.quaternion(*image["qvec"]),
        )
    

class NeuralSurfaceReconstruction():
    def __init__(self, *, img: Path, img_pose: Path, fg_v_sdf: Path, bg_v_sdf: Path):
        self.images = TargetImageDataset(img, img_pose)
        self.fg_V_sdf = torch.load(fg_v_sdf, weights_only=True)
        self.bg_V_sdf = torch.load(bg_v_sdf, weights_only=True)
        self.fg_V_feat = self.fg_V_sdf.repeat(1, 3, 1, 1, 1)  # R,G,B
        self.bg_V_feat = self.bg_V_sdf.repeat(1, 3, 1, 1, 1)  # R,G,B
        self.fg_V_feat[:] = 0.5
        self.bg_V_feat[:] = 0.5

    def query_pixel_sample_points(self, index: int, *,
                                  N_query_points: int,
                                  t_size: float
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        returns:
        1. Nx3 array of target image's pixels (normalized RGB)
        2. NxTx3 array of the pixels' corresponding sample query points in coordinates
        """
        data, pose = self.images[index]

        rot_mat = quaternion.as_rotation_matrix(pose.qvec)   # 3x3 rot mat


        # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h#L715
        # https://calib.io/blogs/knowledge-base/camera-models?srsltid=AfmBOoosTUXUe3QZqSrWoJXC9Yr04axC6Mvx7ru4xjo-yHMRf4H_erhx

        # initial point grid from image information
        px, py = np.mgrid[-pose.width/2:pose.width/2:1.0, -pose.height/2:pose.height/2:1.0]
        pz = np.full(px.shape, 1.0)

        #### FOCAL LENGTH NORMALIZATION (f)
        img_points = np.dstack([(px.flatten()+0.5)/pose.f, (py.flatten()+0.5)/pose.f, pz.flatten()])[0]  # Nx2 points to be rot'd at origin

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
        st = np.transpose(sp, (2, 1, 0))  # NxTx3

        return (target_pixels, st)


    

def initialization():
    nsr = NeuralSurfaceReconstruction(
        img=Path("./data/calculator/raw_images"),
        img_pose=Path("./data/image_pose.json"),
        fg_v_sdf=Path("./data/initial_v_sdf_fg.pt"),
        bg_v_sdf=Path("./data/initial_v_sdf_bg.pt"),
    )





if __name__ == "__main__":
    initialization()

