from torch import float32, int32

itype = int32
ftype = float32
try:
    import pyredner
    device = 'cpu'
except:
    device = 'cuda'