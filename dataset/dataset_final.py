import glob
import os.path as osp
from typing import Callable, Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class BokehDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None):
        self._root_folder = root_folder
        self._transform = transform

        self._source_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._meta_data = self._read_meta_data(osp.join(root_folder, "meta.txt"))
        #self._source_alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.mask_src.png")))
        self._source_alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.227noshreshold306000_src.png")))



    def __len__(self):
        return len(self._meta_data)

    def _read_meta_data(self, meta_file_path: str):
        """Read the meta file containing source / target lens and disparity for each image.

        Args:
            meta_file_path (str): File path

        Raises:
            ValueError: File not found.

        Returns:
            dict: Meta dict of tuples like {id: (id, src_lens, tgt_lens, disparity)}.
        """
        if not osp.isfile(meta_file_path):
            raise ValueError(f"Meta file missing under {meta_file_path}.")

        meta = {}
        with open(meta_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]

            if src_lens == "Canon50mmf1.4BS":
                src_lens = 1.0
            elif src_lens == "Canon50mmf1.8BS":
                src_lens = 0.75
            elif src_lens == "Sony50mmf1.8BS":
                src_lens = 0.625
            elif src_lens == "Sony50mmf1.4BS":
                src_lens = 0.875
            else:
                src_lens = 0.0

            if tgt_lens == "Canon50mmf1.4BS":
                tgt_lens = 1.0
            elif tgt_lens == "Canon50mmf1.8BS":
                tgt_lens = 0.75
            elif tgt_lens == "Sony50mmf1.8BS":
                tgt_lens = 0.625
            elif tgt_lens == "Sony50mmf1.4BS":
                tgt_lens = 0.875
            else:
                tgt_lens = 0.0
            
            disparity = int(disparity)/80

            meta[id] = (src_lens, tgt_lens, disparity)

        return meta

    def __getitem__(self, index):
        
        source = Image.open(self._source_paths[index])
        source_alpha = Image.open(self._source_alpha_paths[index])
        

        filename = osp.basename(self._source_paths[index])
        id = filename.split(".")[0]
        src_lens, tgt_lens, disparity = self._meta_data[id]

        source = self._transform(source)
        source_alpha = self._transform(source_alpha)
        

        if tgt_lens > src_lens :
            # bokeh
            input_cond = np.asarray([src_lens,disparity])
            input_cond = np.float32(input_cond)
            input_cond = torch.from_numpy(input_cond)

            output_cond = np.asarray([tgt_lens-src_lens,disparity])
            output_cond = np.float32(output_cond)
            output_cond = torch.from_numpy(output_cond)

            dis_vector = np.asarray([disparity])
            dis_vector = np.float32(dis_vector)
            dis_vector = torch.from_numpy(dis_vector)

            return {
                "source": source,
                'tgt_blur':True,
                "source_alpha": source_alpha,
                "input_cond": input_cond,
                "output_cond": output_cond,
                "id": id,
                "disparity":dis_vector,
            }

        else:
            #debokeh
            input_cond = np.asarray([src_lens,disparity])
            input_cond = np.float32(input_cond)
            input_cond = torch.from_numpy(input_cond)

            output_cond = np.asarray([src_lens-tgt_lens,disparity])
            output_cond = np.float32(output_cond)
            output_cond = torch.from_numpy(output_cond)

            dis_vector = np.asarray([disparity])
            dis_vector = np.float32(dis_vector)
            dis_vector = torch.from_numpy(dis_vector)

            return {
                "source": source,
                'tgt_blur':False,
                "source_alpha": source_alpha,
                "input_cond": input_cond,
                "output_cond": output_cond,
                "id": id,
                "disparity":dis_vector,
            }