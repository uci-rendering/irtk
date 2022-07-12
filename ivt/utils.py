import torch 

def normalize(vector):
    return vector / torch.norm(vector)

def lookat(origin, target, up):
    dir = normalize(target - origin)
    left = normalize(torch.cross(up, dir))
    new_up = torch.cross(dir, left)

    to_world = torch.zeros(4, 4).to(float)
    to_world[:3, 0] = left
    to_world[:3, 1] = new_up
    to_world[:3, 2] = dir
    to_world[:3, 3] = origin
    to_world[3, 3] = 1.0

    return to_world
