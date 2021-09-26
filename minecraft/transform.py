import torch
import cv2

from torch import Tensor
from numpy import ndarray


def preprocess(img: ndarray) -> Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(img.copy()).float()
