import h5py
import torch
import torch.utils.data as utils_data
import pdb
import numpy as np

class AbstractDataset(utils_data.Dataset):

    def __init__(self, data):
        super(AbstractDataset, self).__init__()
        print('begin to convert to tensor')
        self.data = torch.tensor(data)
        print(type(self.data))
        # self.data = self.data.type(torch.FloatTensor)
        print('over')
    def __getitem__(self, idx):
        return self.data[idx].float()

    def __len__(self):
        return self.data.shape[0]


def get_data_loaders(args):
    phase_list = ['test']
    if args.train:
        phase_list += ['train', 'valid']
    image_shape = None
    with h5py.File(args.path_data, 'r', libver='latest', swmr=True) as f:
        data = {phase: f[phase]['image'][()] for phase in phase_list } 
        image_shape = data[phase_list[0]].shape[-3:]
        print(image_shape)
        dataset = {key: AbstractDataset(val) for key, val in data.items() }
        data_loaders = {
            phase: utils_data.DataLoader(
                val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
            )
            for phase,val in dataset.items()
        }
    return data_loaders, image_shape