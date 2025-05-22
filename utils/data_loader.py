import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class GroupDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        self.sample_ids = self.data["sample_id"].values
        self.groups = self.data["group"].values  

        self.features = self.data.drop(columns=["target", "sample_id", "group"]).values 
        self.labels = self.data["target"].values 

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.sample_ids = torch.tensor(self.sample_ids, dtype=torch.long)
        self.groups = torch.tensor(self.groups, dtype=torch.long)  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.groups[idx], self.sample_ids[idx], idx

    @staticmethod
    def from_dataframe(df):
        obj = GroupDataset.__new__(GroupDataset)
        obj.data = df.copy()

        obj.sample_ids = torch.tensor(df["sample_id"].values, dtype=torch.long)
        obj.groups = torch.tensor(df["group"].values, dtype=torch.long)
        obj.labels = torch.tensor(df["target"].values, dtype=torch.long)
        obj.features = torch.tensor(df.drop(columns=["target", "sample_id", "group"]).values, dtype=torch.float32)

        return obj

def get_column_names(names_file):
    columns = []
    with open(names_file, "r") as file:
        for line in file:
            if ":" in line:  
                col_name = line.split(":")[0].strip()
                columns.append(col_name)
    columns.append("income") 
    return columns

def group_rare_categories(df, col, top_n=5):
    freq = df[col].value_counts()
    categories_to_keep = freq.nlargest(top_n).index
    df[col] = df[col].apply(lambda x: x if x in categories_to_keep else "Other")
    return df

#########################################################################################################

def load_data(config):
    dataset = config["dataset"].lower()

    ######## Dataset #########
    train_file = 0
    val_file = 0
    test_file = 0
    
    ##########################
    train_df = pd.read_csv(train_file)
    
    train_dataset = GroupDataset(train_file)
    valid_dataset = GroupDataset(val_file)
    test_dataset = GroupDataset(test_file)

    batch_size = config.get("train_params", {}).get("batch_size", 32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader, train_df


def dataloader_to_numpy(dataloader):
    all_x, all_y, all_groups = [], [], []

    for batch in dataloader:
        x, y, group, *_ = batch 
        all_x.append(x)
        all_y.append(y)
        all_groups.append(group)

    X = torch.cat(all_x).cpu().tolist()
    y = torch.cat(all_y).cpu().tolist()
    groups = torch.cat(all_groups).cpu().tolist()
    
    return X, y, groups
