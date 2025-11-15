import numpy as np
import pandas as pd
import random

from torch.utils.data import Dataset

class Mobile_Price(Dataset):
    def __init__(self, data_path, split, ratio=0.8):
        assert split in ["train", "test"], "split must be 'train' or 'test'!"

        raw_data = pd.read_csv(data_path)

        data_np = np.array(raw_data)

        num_data, self.num_feat = data_np.shape

        num_train = int(num_data*ratio)

        random.seed(42)
        train_id = random.sample(range(num_data), num_train)
        test_id = [i for i in list(range(num_data)) if i not in train_id]

        train_data = data_np[train_id, :]
        test_data = data_np[test_id, :]

        if split == "train":
            self.data2use = train_data
        else:
            self.data2use = test_data

    def __getitem__(self, idx):
        feat = self.data2use[idx, :-1]
        label = self.data2use[idx, -1]

        return feat, label

    def __len__(self):
        return len(self.data2use)
    
    @property
    def get_num_feat(self):
        return self.num_feat - 1
    
class Wine_QT(Dataset):
    def __init__(self, data_path,split,ratio = 0.8):

        # sets seed: internal seed value ensures agreement between test and train use of class
        random.seed(42)

        # requires one of list of acceptable arguments
        assert split in ["train","test"], "split parameter must have 'train' or 'test' for argument"

        # reads in the data
        self.raw_data = pd.read_csv(data_path)

        names = self.raw_data.columns

        if 'Id' in names:
            self.raw_data.drop('Id', axis=1, inplace=True)
            self.raw_data.to_csv('WineQT.csv', index= False)

        
        # converts data to numpy object
        self.data_np = np.array(self.raw_data)

        ##### updates classes for proper loss values #####
        # gets unique classes
        classes = np.unique(self.data_np[:, -1])
        # makes dictionary for classes using index as value
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Replace the last column with the mapped labels
        self.data_np[:, -1] = np.vectorize(class_to_idx.get)(self.data_np[:, -1]).astype(int)


        # gets data shape
        self.num_data, self.num_feat = self.data_np.shape

        ##### allocates data to test/train #####
        # establishes length of training data
        num_train = int(self.num_data*ratio)

        # randomly allocates indexes for splitting test and train data
        train_idx = random.sample(range(self.num_data), num_train)
        test_idx = [i for i in list(range(self.num_data)) if i not in train_idx]

        # creates test and train subsets  
        train_data = self.data_np[train_idx, :]
        test_data = self.data_np[test_idx, :]

        # allows class used for both testing/training purposes
        if split == "train":
            self.data2use = train_data
        else:
            self.data2use = test_data

    def __getitem__(self, idx):
        feat = self.data2use[idx, :-1]
        label = self.data2use[idx, -1]

        return feat, label

    def __len__(self):
        return len(self.data2use)

    @property
    def get_num_feat(self):
        return self.num_feat - 1
    
    @property
    def get_num_data(self):
        return self.get_num_data
    
    @property
    def get_full_dataset(self):
        return self.raw_data
    
    @property
    def get_class_values(self):
        unique_classes = np.sort(self.raw_data.iloc[:,-1].unique())
        return unique_classes
    