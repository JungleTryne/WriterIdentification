import os
import pathlib
from typing import Dict, Any
import cv2 as cv

from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CVLHandwritingWordsDataset(Dataset):
    """
    Dataset of handwritten words, taken from CVL Database
    """

    def __init__(self, root_dir: pathlib.Path, debug: bool=True):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

        if debug:
            self.files = self.files[:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = os.path.join(self.root_dir, self.files[idx])
        return {
            "image_tensor": cv.imread(image_path, 0),
            "author_id": image_path.split('/')[-1].split('-')[0],
            "text_id": image_path.split("/")[-1].split("-")[1],
        }

    def get_distribution(self):
        result = dict()
        for file in self.files:
            author_id = file.split('-')[0]
            result[author_id] = 1 if author_id not in result else result[author_id] + 1
        return result