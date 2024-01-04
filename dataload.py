import pandas as pd
import numpy as np
import torch
import os

from torch.utils import data
from torch.utils.data import Dataset
import json


class MagnaTagATune(data.Dataset):
    def __init__(self, dataset_key , dataset, global_min=None, global_max=None):
        """
        Given the dataset path, create the MagnaTagATune dataset. Creates the
        variable self.dataset which is a list of 3-element tuples, each of the
        form (filename, samples, label):
            1) The filename which a given set of audio samples belongs to
	        2) The audio samples which relates to a 29.1 second music clip
               resampled to 12KHz. Each array is of shape: [10, 1, 34950].
               Where 10 represents the sub-clips, 1 is the channel dim and
               34950 are the number of samples in the sub-clip.
	        3) The multiclass label of the audio file of shape: [50].
        Args:
            dataset_path (str): Path to train_labels.pkl or val_labels.pkl
        """
        print(f"Loading data from {dataset_key}_dataset")

        dataset_path = dataset[dataset_key]["annotations_path"]
        samples_path = dataset[dataset_key]["samples_path"]

        self.dataset = pd.read_pickle(dataset_path)
        self.samples_path = samples_path

        self.global_min = global_min
        self.global_max = global_max

    def __getitem__(self, index):
        """
        Given the index from the DataLoader, return the filename, spectrogram,
        and label
        Args:
            index (int): the dataset index provided by the PyTorch DataLoader.
        Returns:
            filename (str): the filename of the .wav file the spectrogram
                belongs to.
            samples (torch.FloatTensor): the audio samples of a 29.1
                second audio file.
            label (toRch.FloatTensor): the class of the file/audio samples.
        """


        data = self.dataset.iloc[index]

        filename = data['file_path']
        filepath = os.path.join(self.samples_path, filename)
        if not os.path.exists(filepath):
        # Handle missing file (return zeros, skip, etc.)
            return torch.zeros(1), torch.zeros(1)

        samples = torch.from_numpy(np.load(f"{self.samples_path}/{filename}"))

        label = torch.FloatTensor(data['label'])
        samples = samples.view(10, -1).contiguous() # Create 10 subclips

        if self.global_min is not None and self.global_max is not None:
            samples = 2 * ((samples - self.global_min) / (self.global_max - self.global_min)) - 1

        return filename, samples.unsqueeze(1), label

    def __len__(self):
        """
        Returns the length of the dataset (length of the list of 4-element
            tuples). __len()__ always needs to be defined so that the DataLoader
            can create the batches
        Returns:
            len(self.dataset) (int): the length of the list of 4-element tuples.
        """
        return self.dataset.shape[0]

class SpecMagnaTagATune(Dataset):
    def __init__(self, dataset_key, dataset):

        print(f"Loading data from {dataset_key}_dataset")

        dataset_path = dataset[dataset_key]["annotations_path"]
        samples_path = dataset[dataset_key]["samples_path"]

        self.dataset = pd.read_pickle(dataset_path)
        self.samples_path = samples_path

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        filename = data['file_path']

        filepath = os.path.join(self.samples_path, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return torch.zeros(1, 128, 1024), torch.zeros(50)

        spec_file = np.load(filepath)
        spec_tensor = torch.from_numpy(spec_file)

        label = torch.FloatTensor(data['label'])

        return filename, spec_tensor, label

    def __len__(self):
        return len(self.dataset)
    

 
