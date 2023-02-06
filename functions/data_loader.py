import torch
from torch.utils.data import Dataset
from typing import Callable,Optional
import os
import numpy as np

class Holo_Recon_Dataloader(Dataset):

    def __init__(self,
                 root: str,
                 data_type:str,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 return_distance: bool=None,
                 return_path: bool=None,
                 pth_from_pickle: bool=None,
                 pickle_pth: str=None,
                 ratio: float=None,
    ) -> None:

        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.data_type = data_type
        self.return_distance = return_distance
        self.return_path = return_path
        self.pth_from_pickle = pth_from_pickle
        self.pickle_pth = pickle_pth
        self.data_list=[]
        self.root_list=[]

        if self.pth_from_pickle:
            import pickle
            with open(self.pickle_pth, 'rb') as fr:
                user_load = pickle.load(fr)

            for i in user_load.values():
                self.data_list.extend(i)

        else:
            data_root = os.path.join(root, image_set, self.data_type)
            self.data_list = [[os.path.join(i, j) for j in os.listdir(os.path.join(data_root, i))] for i in os.listdir(data_root)]
            self.data_list = np.ravel(np.array(self.data_list))

        np.random.shuffle(self.data_list)

        if ratio is not None:
            dat_num = len(self.data_list)

            self.ratio_idx_set = np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[self.ratio_idx_set]

        self.data_num = len(self.data_list)

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        if self.pth_from_pickle:
            pth = self.data_list[index]
        else:
            pth = os.path.join(self.root, self.image_set, self.data_type, self.data_list[index])

        data = self.load_matfile(pth)
        holo = data['holography']

        if self.return_distance:
            distance = data['distance']
            if self.transform is not None:
                holo = self.transform(holo)
                distance = torch.from_numpy(distance)

            if self.return_path:
                return holo, distance, pth

            return holo, distance
        else:
            if self.transform is not None:
                holo = self.transform(holo)

            if self.return_path:
                return holo, pth

            return holo


    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data

