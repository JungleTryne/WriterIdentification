import pathlib
from typing import Dict, Any
import cv2

from torch.utils.data import Dataset
from utils.globals import DATASET_HEIGHT
from utils.images_utils import image_resize, convert_image

import os


class CVLHandwritingLinesDataset(Dataset):
    """
    Dataset of handwritten lines, taken from CVL Database
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
        image_tensor = cv2.imread(image_path, -1)
        
        image_tensor = image_resize(image_tensor, height = DATASET_HEIGHT)
        image_tensor = convert_image(image_tensor)
        
        sample_info = image_path.split('/')[-1].split(".")[0].split('-')
        author_id = sample_info[0]
        text_id = sample_info[1]
        line_id = sample_info[2]
        
        return {
            "image_tensor": image_tensor,
            "author_id": int(author_id),
            "text_id": int(text_id),
            "line_id": int(line_id),
        }