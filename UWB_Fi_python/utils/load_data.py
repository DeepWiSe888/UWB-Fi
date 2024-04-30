import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data(data_dir, batch_size,shuffle):


    with h5py.File(data_dir, 'r') as f:
        x_data = f['input'][()]
        y_data = f['output'][()]
    # expand dim---channel is 1
    # x_data = np.expand_dims(x_data, axis=1).astype(np.float32)
    x_data = x_data.astype(np.float32)
    y_data = np.expand_dims(y_data, axis=1).astype(np.float32)
    
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 4,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    # simple statistics of output data 输出一个简单的数据统计值
    y_data_mean = np.mean(y_data, 0)
    y_data_var = np.sum((y_data - y_data_mean) ** 2)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats
