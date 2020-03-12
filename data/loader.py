'''
@Author: Qin Yang
@Date: 2020-03-12 18:02:04
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 21:45:21
'''
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

__all__ = ['gen_data_loader']


class ECGDataset(Dataset):
    def __init__(self, file_path):
        super(ECGDataset, self).__init__()
        self._load_h5py_file(file_path)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

    def _load_h5py_file(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.data = np.asarray(f['data'], dtype=np.float32)
            self.label = np.asarray(f['label'], dtype=int)


def gen_data_loader(file_path, batch_size=32, num_workers=4, pin_memory=False):
    ecg_dataset = ECGDataset(file_path)
    ecg_dataloader = DataLoader(
        ecg_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory)
    return ecg_dataloader
