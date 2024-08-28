from dataclasses import dataclass

@dataclass
class Image:
    file_name: str
    width: int
    height: int
    f: float  # focal length
    k: float  # radial distortion
    tvec: list[float]  # translation vector
    qvec: list[float]  # quaternion rot vector => euler's rot vector?
