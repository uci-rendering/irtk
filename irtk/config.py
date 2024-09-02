import torch
import numpy as np
from typing import List, Tuple, Union

configs = {
    "itype": torch.int32,
    "ftype": torch.float32,
    "device": 'cuda',
}

TensorLike = Union[torch.Tensor, np.ndarray, List, Tuple]