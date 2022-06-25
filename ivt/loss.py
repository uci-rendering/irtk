import torch

def l1_loss(target_image, rendered_image):
    return (target_image - rendered_image).abs().mean()
