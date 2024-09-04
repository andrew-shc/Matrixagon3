import torch


def display_mem(t: torch.Tensor) -> str:
    return str(t.untyped_storage().nbytes()/1000**2)+" MB"
