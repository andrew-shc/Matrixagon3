import torch
from torch import nn

from initialization import VoxelSector


class NeuralDistillationModel(nn.Module):
    def __init__(self, vertices: torch.Tensor, normal: torch.Tensor):
        super().__init__()

        self.vertices = vertices
        self.normal = normal

        self.albedo = nn.Parameter()  # per vertex
        self.roughness = nn.Parameter()  # per vertex
        self.sg = nn.Parameter()

    def L_i(self, vert: torch.Tensor, ω_i: torch.Tensor) -> torch.Tensor:
        """
        Vx3 (XYZ), Vx2 -> Vx3 (RGB)
        vertex, corresponding light direction (ω_i ∈ Ω) => indirect illumination
        """
        pass

    def BRDF(self, vert: torch.Tensor, ω_o: torch.Tensor, ω_i: torch.Tensor) -> torch.Tensor:
        """
        Vx3, Vx2, Vx256x2 -> Vx3

        vertex, incoming light, outgoing radiance, normal => color
            + albedo & roughness
            ~ n (normal)                    self.normal
            ~ l (light directions)          ω_i / 256 pre-sample light dirs. (16x16) per vert
                importance sampling
            ~ v (viewer direction)          ω_o
            ~ h (angle bisector of v&l)     halfway between l & v

        Lambertian diffuse * (1-Fresnel) + Cook-Torrance
        d = D(h)
        f = F(v,h)
        g = G(l,v,h)
        (1-f) * albedo[v]/pi + f * dg/(4*nl*nv)
        """
        D = h
        F = v,h
        G = l,v,h

        return D*F*G/(4*(self.normal*ω_i)*(self.normal*ω_o))

    def forward(self, vert: torch.Tensor, ω_o: torch.Tensor):
        predicted_lo = torch.tensor()

        predicted_lo = self.L_i() * self.BRDF() * torch.dot(self.normal, ω)




class NeuralDistillationLoss(nn.Module):
    def __init__(self, teacher_lo: nn.Module):
        super().__init__()

    def forward(self):
        pass


@torch.no_grad
def marching_cube(sdf: torch.Tensor) -> torch.Tensor:
    """
    using marching cube: voxel point grid -> mesh
    """
    pass

def compute_normal(v: torch.Tensor) -> torch.Tensor:
    """
    vertex -> corresponding normal vector direction (S^2)
    """
    pass
    

class NeuralDistllation:
    def __init__(self, fg: VoxelSector, lo: nn.Module):
        pass




def distillation():
    pass


if __name__ == "__main__":
    distillation()
