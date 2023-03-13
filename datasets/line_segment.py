from random import randint

import cv2
import cv2 as cv

import os
import pathlib

import torch
from torch.utils.data import Dataset

from typing import Dict, Any


class LineSegmentsDataset(Dataset):
    """
    Dataset of handwritten fragments, taken from CVL Database
    """

    def __init__(self, root_dir: pathlib.Path, debug: bool=False):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

        if debug:
            self.files = self.files[:300]
            
        self.authors_indexes = dict()
        from_idx, to_idx = 0, 0
        current_author_id = None
        self.files = sorted(self.files)
        
        for i, file in enumerate(self.files):
            desc = self._get_file_description(file)
            if desc["author_id"] != current_author_id:
                if current_author_id is not None:
                    self.authors_indexes[current_author_id] = (from_idx, i)
                    from_idx = i
                    
                current_author_id = desc["author_id"]
            to_idx = i + 1
            
        self.authors_indexes[current_author_id] = (from_idx, to_idx)
        self.pregenerated = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = os.path.join(self.root_dir, self.files[idx])
        
        image_tensor = cv.imread(image_path, -1)
        width = image_tensor.shape[1]
        height = image_tensor.shape[0] * 2
        dim = (width, height)
        image_tensor = cv2.resize(image_tensor, dim, interpolation = cv2.INTER_LINEAR)
        
        image_tensor = torch.Tensor(image_tensor)[None, :, :]
        image_tensor = image_tensor.repeat(3, 1, 1)
        
        return {
            "image_tensor": image_tensor,
            **self._get_file_description(image_path)
        }
    
    def _get_file_description(self, file_path):
        sample_info = file_path.split('/')[-1].split(".")[0].split('-')
        
        author_id = sample_info[0]
        text_id = sample_info[1]
        line_id = sample_info[2]
        segment_id = sample_info[3]
        
        return {
            "author_id": int(author_id),
            "text_id": int(text_id),
            "line_id": int(line_id),
            "segment_id": int(segment_id)
        }
    
    def get_random_pair(self, author_id: int, can_link: bool):
        from_idx, to_idx = self.authors_indexes[author_id]
        if can_link:
            pair_idx = randint(from_idx, to_idx - 1)
            return self.__getitem__(pair_idx)
        else:
            assert len(self.files) - (to_idx - from_idx) - 1 > 0, "Cannot generate cannot_link"
            pair_idx = randint(0, len(self.files) - (to_idx - from_idx) - 1)
            if pair_idx >= from_idx:
                pair_idx += (to_idx - from_idx)
            return self.__getitem__(pair_idx)
            