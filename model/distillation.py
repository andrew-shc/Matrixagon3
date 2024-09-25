import torch
from torch import nn
from torch.nn import functional as F

from torchmcubes import marching_cubes

from initialization import VoxelSector





@torch.no_grad
def compute_mesh(sdf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    using marching cube: voxel point grid -> mesh (vertices, faces)
    """
    verts, faces = marching_cubes(sdf, thresh=0.0)  # 0-level sdf

    # removes duplicate vertices
    verts, reverse_ind = torch.unique(verts, dim=0, return_inverse=True)
    faces = reverse_ind[faces]
    
    return verts, faces

@torch.no_grad
def compute_normal(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    vertex (Vx3) -> corresponding normal vector direction (S^2)
    """
    # Fx(A,B,C) -> Fx(A,B,C)x(X,Y,Z)
    vf = verts[faces]
    face_norm = torch.linalg.cross(vf[:,1]-vf[:,0], vf[:,2]-vf[:,0])

    # index_put_: supports simulataneous summation for duplicate indices in faces
    vert_norm = torch.zeros(verts.shape, device="cuda")
    vert_norm.index_put_(indices=(faces.flatten(),),
                        values=torch.repeat_interleave(face_norm, 3, dim=0),
                        accumulate=True)
    vert_norm = F.normalize(vert_norm)  # (Nx3)

    # convert normals from xyz vector into spherical angle
    spherical = torch.stack([
        # THETA
        torch.acos(vert_norm[:,1] / torch.linalg.norm(vert_norm, dim=1)),  #*(180.0/np.pi)
        # PHI
        torch.sign(vert_norm[:,2])*torch.acos(vert_norm[:,0] / torch.linalg.norm(vert_norm[:,[0,2]], dim=1)),   #*(180.0/np.pi)
    ], dim=1)

    return spherical




class NeuralDistillationModel(nn.Module):
    def __init__(self, vertices: torch.Tensor, normal: torch.Tensor, device: torch.device):
        super().__init__()

        self.vertices = vertices
        self.normal = normal

        self.albedo = nn.Parameter(torch.ones(self.vertices.shape[0]))  # per vertex
        self.roughness = nn.Parameter(torch.ones(self.vertices.shape[0]))  # per vertex
        self.sg = nn.Parameter(torch.ones((256, 6), device=device))  # 256x[R,G,B,λ,θp,φp]

    def L_i(self, vert: torch.Tensor, ω_i: torch.Tensor) -> torch.Tensor:
        """
        Vx3 (XYZ), Vx2 -> Vx3 (RGB)
        vertex, corresponding light direction (ω_i ∈ Ω) => indirect illumination
        """
        pass

    def L_SG_env(self, ω_i: torch.Tensor) -> torch.Tensor:
        """
        incoming light direction (?x2) => scalar RGB value
        """
        θp, φp = self.sg[:, 4], self.sg[:, 5]
        spherical_dot = lambda θv, φv: torch.sin(θv)*torch.sin(θp)*torch.cos(φv-φp) + torch.cos(θv)*torch.cos(θp)
        value_per_lobe = lambda θv, φv: self.sg[:, 0:3]*torch.exp(self.sg[:, 3]*(spherical_dot(θv, φv)-1))[:, torch.newaxis]

        return value_per_lobe(ω_i[0], ω_i[1]).sum(axis=0)

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
    def __init__(self, teacher_lo: nn.Module, device: torch.device):
        super().__init__()

    def forward(self):
        pass

    

class NeuralDistllation:
    def __init__(self, fg: VoxelSector, lo: nn.Module, device: torch.device):
        pass




def distillation():
    pass


if __name__ == "__main__":
    distillation()
