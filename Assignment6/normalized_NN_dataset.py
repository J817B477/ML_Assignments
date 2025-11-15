import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset

class Wine_QT(Dataset):
    def __init__(self, data_path, split, ratio=0.8):
        random.seed(42)
        assert split in ["train", "test"], "split must be 'train' or 'test'!"

        self.raw_data = pd.read_csv(data_path)

        # drop 'Id' column if exists
        if 'Id' in self.raw_data.columns:
            self.raw_data.drop('Id', axis=1, inplace=True)
        
        # encode class labels
        classes = np.unique(self.raw_data.iloc[:, -1])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        self.raw_data.iloc[:, -1] = self.raw_data.iloc[:, -1].map(class_to_idx)

        # convert to numpy
        self.data_np = np.array(self.raw_data, dtype=float)

        # split indices
        num_data = len(self.data_np)
        num_train = int(num_data * ratio)
        train_idx = random.sample(range(num_data), num_train)
        test_idx = [i for i in range(num_data) if i not in train_idx]

        # extract train/test subsets
        train_data = self.data_np[train_idx, :]
        test_data = self.data_np[test_idx, :]

        # compute normalization stats from training data only
        self.feat_mean = train_data[:, :-1].mean(axis=0)
        self.feat_std = train_data[:, :-1].std(axis=0)

        # normalize features using train stats
        train_data[:, :-1] = (train_data[:, :-1] - self.feat_mean) / self.feat_std
        test_data[:, :-1] = (test_data[:, :-1] - self.feat_mean) / self.feat_std

        # select split
        self.data2use = train_data if split == "train" else test_data

    def __getitem__(self, idx):
        feat = self.data2use[idx, :-1]
        label = int(self.data2use[idx, -1])
        return feat, label

    def __len__(self):
        return len(self.data2use)

    @property
    def get_num_feat(self):
        return self.data2use.shape[1] - 1
    
    @property
    def get_class_values(self):
        return np.sort(np.unique(self.data_np[:, -1]))

if __name__ == "__main__":
    pass