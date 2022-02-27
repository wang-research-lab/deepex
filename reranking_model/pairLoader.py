import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import csv
import os
from os.path import join

import pdb

class Mydataset(Dataset):
    def __init__(self, dataset, file_name):
        super(Mydataset, self).__init__()
        with open(join("../data", dataset, file_name + ".csv"), "r", encoding="utf-8") as f:
            csv_reader_rows = csv.reader(f)

            self.x_train = []
            self.y_train = []

            num = 0
            for row in csv_reader_rows:
                text = row[0]
                # maxium 512 tokens
                if len(text) > 400:
                    text_list = text[:400].split(" ")
                    # truncate the too long text, actually test_list here is ordered word list
                    text_list = text_list[:-1]
                    text = " ".join(text_list)
                try:
                    self.x_train.append([text+'[SEP]'+row[1]])
                    self.y_train.append([1]) # actually don't need label here
                    num += 1
                except:
                    continue
                
            self.num = num

    # indexing
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num


