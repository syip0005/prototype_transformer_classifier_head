from torch.utils.data import Dataset
import pandas as pd
import os
from random import shuffle
import json
import csv
import torch


class AG_NEWS_TOKENIZED(Dataset):

    def __init__(self, root_dir='./data/AG_NEWS_MOD/', train=True, tokenizer=None, N=None):

        """
        Custom dataset for AG NEWS

        Args:
            root_dir (string): Directory with all the files
            transform (callable, optional): Transform to be applied
        """

        # TODO: change varibable name from type (reserved)

        if train:
            file_path = root_dir + 'train.csv'
        else:
            file_path = root_dir + 'test.csv'

        self.text = pd.read_csv(file_path)

        # If N is selected, pick random list
        if N is not None:

            if N > len(self.text):

                raise ValueError("Cannot exceed dataset size of {}".format(len(self.text)))

            else:

                self.text = self.text.sample(frac=(N / len(self.text))).reset_index(drop=True)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        sentence = self.text.iloc[idx, 2]
        label = self.text.iloc[idx, 0] - 1  # to zero index it

        if self.tokenizer:  # use BertTokenizer here
            sentence = self.tokenizer(sentence, padding="max_length", truncation=True, return_tensors='pt')

        return {key: value.squeeze() for key, value in sentence.items()}, label  # squeeze for correct input shape


class EURLEX57K(Dataset):

    # TODO: write doc string
    # TODO: change varibable name from type (reserved)

    def __init__(self, root_dir='./data/EURLEX57K', type='Train', tokenizer=None, N=None):

        type = type.lower()
        valid_types = ['train', 'test', 'dev']

        if type not in valid_types:
            raise ValueError('Please use only {}'.format(', '.join(valid_types)))
        else:
            self.file_path = root_dir + '/' + type + '/'

        # List all files in the directory in alphabetical
        self.all_files = sorted(os.listdir(self.file_path))

        if N is not None:

            if N > len(self.all_files):
                raise ValueError("Cannot exceed dataset size of {}".format(len(self.text)))

            else:
                shuffle(self.all_files)
                self.all_files = self.all_files[:N]

        self.tokenizer = tokenizer

        # Read index converter into csv
        # for conversions of ridiculous concept numbers to normalised 0-index
        self.converter = dict()
        with open(root_dir + '/' + 'index_converter.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.converter[int(row[0])] = int(row[1])

    def __len__(self):

        return len(self.all_files)

    def __getitem__(self, idx):

        file_to_retrieve = self.all_files[idx]

        path = self.file_path + file_to_retrieve

        with open(path) as f:
            file_dict = json.load(f)

            sentence = ' '.join(file_dict['main_body'])
            labels = [self.converter[int(i)] for i in file_dict['concepts'] if
                      i.isdigit()]  # Convert them to normalised indicies

        # TODO: reconsider simple truncation for this dataset
        if self.tokenizer:  # use BertTokenizer here
            sentence = self.tokenizer(sentence, padding="max_length", truncation=True, return_tensors='pt')

        # Turn labels into tensors (hard-code in number of labels)
        label_tensor = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=4270).sum(dim=0)

        return {key: value.squeeze() for key, value in sentence.items()}, label_tensor.float()

