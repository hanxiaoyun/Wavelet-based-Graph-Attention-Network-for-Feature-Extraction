

import torch

from torch.utils.data import Dataset
import numpy as np


def get_adjacent_matrix(distance_file: str) -> np.array:

    A = np.load(distance_file)

    return A

def get_flow_data(flow_file: str) -> np.array:

    data = np.load(flow_file)

    flow_data = data.transpose([1, 0, 2])[:, :, -1][:, :, np.newaxis]


    return flow_data




class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):


        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]
        self.test_days = divide_days[1]
        self.history_length = history_length
        self.time_interval = time_interval

        self.one_day_length = 1

        self.graph = get_adjacent_matrix(distance_file=data_path[0])

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1) # self.flow_norm为归一化的基

    def __len__(self):

        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):

        if self.train_mode == "train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x)
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):

        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):


        norm_base = LoadData.normalize_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):

        max_data = np.max(data, norm_dim, keepdims=True)
        min_data = np.min(data, norm_dim, keepdims=True)

        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):

        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data,ty):

        mid = min_data
        base = max_data - min_data
        if ty=='pre':
            recovered_data = data * base + mid
        else:
            recovered_data = data * base + mid
        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


