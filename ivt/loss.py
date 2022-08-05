import torch
from ivt.io import write_exr

def l1_loss(target_image, rendered_image):
    return (target_image - rendered_image).abs().mean()
