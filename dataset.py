# https://github.com/NVlabs/edm/blob/main/dataset_tool.py

import os
from PIL import Image
from torch.utils.data import Dataset
import glob
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
from tqdm import tqdm
from utils import NoiseGenerator, store_uint128_pairs, load_uint128_pairs
import json
import zipfile
import csv
class DatasetWithLatentCond(Dataset):
    def __init__(self, im_dir, latent_dir, input_nc, label_dim):
        super().__init__()
        self.label_dim = label_dim
        im_dir = os.path.abspath(im_dir)
        latent_dir = os.path.abspath(latent_dir)
        self.input_nc = input_nc
        self.im_dir = im_dir
        self.latent_dir = latent_dir
        self._zipfile_im = None
        self._zipfile_latent = None
        print(f"im_dir = {im_dir}, latent_dir = {latent_dir}")


        if os.path.isdir(im_dir) or os.path.isdir(latent_dir):
            assert os.path.isdir(im_dir) and os.path.isdir(latent_dir), "Both im_dir and latent_dir must be directories"
            self._type = 'dir'
            # Path to the pre-cached file list
            cached_list_path = os.path.join(latent_dir, 'cached_file_list.txt')

            # Check if pre-cached file list exists
            if os.path.exists(cached_list_path):
                print(f"Loading pre-cached file list from {cached_list_path}")
                # Load pre-cached file list
                with open(cached_list_path, 'r') as file:
                    latent_list = [line.strip() for line in file]
            else:
                # Generate file list using os.walk
                latent_list = []
                for root, dirs, files in os.walk(latent_dir):
                    for file in tqdm(files, desc=f"Processing {root}"):
                        if file.endswith('.npy') or file.endswith('.npz'):
                            abs_path = os.path.join(root, file)
                            latent_list.append(os.path.relpath(abs_path, start=latent_dir))
                latent_list.sort()

                # Cache the file list
                with open(cached_list_path, 'w') as file:
                    for path in latent_list:
                        file.write(path + '\n')
        elif self._file_ext(im_dir) == '.zip' or self._file_ext(latent_dir) == '.zip':
            assert self._file_ext(im_dir) == '.zip' and self._file_ext(latent_dir) == '.zip', "Both im_dir and latent_dir must be zip files"
            self._type = 'zip'
            latent_list = self._get_zipfile_latent().namelist()
            latent_list.sort()
        else:
            raise ValueError("Unsupported file type for im_dir or latent_dir")
        self.latent_list = latent_list
        self.im_names = [self._get_image_name(latent_name) for latent_name in latent_list]
        print(f"len(self.latent_list) = {len(self.latent_list)}, len(self.im_names) = {len(self.im_names)}")
        self.noise_gen = NoiseGenerator(0)

        # Define transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if input_nc == 3 else transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        if label_dim == 0:
            self.label_dict = None
        else:
            self.label_dict = {}
            dict_path = os.path.split(im_dir)[0]
            dict_path = os.path.join(dict_path, 'images_labels.csv')
            with open(dict_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    im_relative_path, cls_label = row[0], int(row[1])
                    self.label_dict[im_relative_path] = cls_label
            # Preview keys and vals
            for i in range(5):
                print(f"key = {list(self.label_dict.keys())[i]}, val = {list(self.label_dict.values())[i]}")
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    def _remove_ext(self, fname):
        return os.path.splitext(fname)[0]

    def _get_zipfile_im(self):
        assert self._type == 'zip'
        if self._zipfile_im is None:
            self._zipfile_im = zipfile.ZipFile(self.im_dir)
        return self._zipfile_im

    def _get_zipfile_latent(self):
        assert self._type == 'zip'
        if self._zipfile_latent is None:
            self._zipfile_latent = zipfile.ZipFile(self.latent_dir)
        return self._zipfile_latent
    def __getstate__(self):
        state = dict(self.__dict__)
        state['_zipfile_im'] = None
        state['_zipfile_latent'] = None
        return state
    def _get_image_name(self, latent_name):
        fname = self._remove_ext(latent_name)
        return fname + '.png'
    def __len__(self):
        return len(self.latent_list)
    def _load_raw_image(self, idx):
        fname = self.im_names[idx]
        if self._type == 'dir':
            with open(os.path.join(self.im_dir, fname), 'rb') as file:
                img = Image.open(file)
                if self.input_nc == 1:
                    img = img.convert('L')
                img = np.array(img)
                return img
        elif self._type == 'zip':
            with self._get_zipfile_im().open(fname, 'r') as file:
                img = Image.open(file)
                if self.input_nc == 1:
                    img = img.convert('L')
                img = np.array(img)
                return img
    def _load_raw_state(self, idx):
        if self._type == 'dir':
            return load_uint128_pairs(os.path.join(self.latent_dir, self.latent_list[idx]))
        elif self._type == 'zip':
            with self._get_zipfile_latent().open(self.latent_list[idx], 'r') as file:
                data = file.read()
                # Split the data back into two uint128 integers
                a_bytes, b_bytes = data[:16], data[16:]
                a, b = int.from_bytes(a_bytes, byteorder='big'), int.from_bytes(b_bytes, byteorder='big')
                return np.array([[a, b]])
    def close(self):
        try:
            if self._zipfile_im is not None:
                self._zipfile_im.close()
            if self._zipfile_latent is not None:
                self._zipfile_latent.close()
        finally:
            self._zipfile_im = None
            self._zipfile_latent = None
    def __getitem__(self, idx):
        img = self._load_raw_image(idx)
        img = self.transforms(img)
        state = self._load_raw_state(idx)
        latent, _ = self.noise_gen.sample_noise(img.unsqueeze(0).shape, state)
        latent = torch.tensor(latent, dtype=torch.float32).squeeze(0)
        
        
        if self.label_dict is not None:
            label = self.label_dict[self.im_names[idx]] 
            label = torch.tensor(label, dtype=torch.int64)
            label_onehot = torch.zeros(self.label_dim)
            label_onehot[label] = 1
        else:
            label_onehot = torch.zeros(1)
        return img, latent, label_onehot, self._remove_ext(self.im_names[idx])