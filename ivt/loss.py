import torch
from ivt.io import write_exr

def l1_loss(target_image, rendered_image):
    loss = 0
    I = rendered_image
    T = target_image
    diff = (I - T)
    diff = diff.nan_to_num(nan=0)
    diff = diff.abs()
    loss += diff.sum()
    return loss

    # return (target_image - rendered_image).abs().mean()
