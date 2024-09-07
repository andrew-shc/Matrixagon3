import torch
from torch import nn

from initialization import VoxelSector


class NeuralDistillationModel(nn.Module):
    def __init__(self):
        self.albedo = torch.tensor()
        self.roughness = torch.tensor()
        self.sg = torch.tensor()

    def forward(self, v: torch.Tensor, ω: torch.Tensor):
        predicted_lo = torch.tensor()




class NeuralDistillationLoss(nn.Module):
    def __init__(self, teacher_lo: nn.Module):
        pass

    def forward(self):
        pass



class NeuralDistllation:
    def __init__(self, fg: VoxelSector, lo: nn.Module):
        pass

    @staticmethod
    def marching_cube(v: torch.Tensor) -> torch.Tensor:
        """
        using marching cube: voxel point grid -> mesh
        """
        pass



def distillation():
    pass


if __name__ == "__main__":
    distillation()
