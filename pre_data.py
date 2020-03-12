'''
@Author: Qin Yang
@Date: 2020-03-12 21:39:34
@Email: qinyangforever@foxmail.com
@LastEditors: Qin Yang
@LastEditTime: 2020-03-12 21:43:01
'''
import h5py
import numpy as np


def main():
    data = np.random.randn(1000, 12, 1024)
    label = np.random.randint(0, 1, size=(1000,), dtype=int)

    with h5py.File('./data/raw/test.h5py') as f:
        f.create_dataset(
            name='data',
            data=data,
            dtype=float
        )
        f.create_dataset(
            name='label',
            data=label,
            dtype=int
        )


if __name__ == "__main__":
    main()
