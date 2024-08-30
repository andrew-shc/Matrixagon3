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


def initialization():
    target_image_dataset = TargetImageDataset(
        Path("./data/calculator/raw_images"),
        Path("./data/image_pose.json")
    )

    fg_V_sdf = torch.load("./data/initial_v_sdf_fg.pt", weights_only=True)
    bg_V_sdf = torch.load("./data/initial_v_sdf_bg.pt", weights_only=True)

    print(fg_V_sdf)
    print(bg_V_sdf)



if __name__ == "__main__":
    initialization()

