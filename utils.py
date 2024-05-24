import torch
import numpy as np
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import json 
from collections import defaultdict
from torchvision.utils import save_image



def straightness(traj, mean = True):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        if mean:
            # Average along the batch dimension
            mse.append(torch.mean((v - base) ** 2))
        else:
            # Average except the batch dimension
            if len(v.shape) == 2:
                mse.append(torch.mean((v - base) ** 2, dim = 1))
            elif len(v.shape) == 4:
               mse.append(torch.mean((v - base) ** 2, dim = [1, 2, 3]))
    mse = torch.stack(mse)
    if mean:
        return torch.mean(mse)
    else:
        return torch.mean(mse, dim = 0)

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_traj(traj, path):
    traj = torch.cat(traj, dim=3)
    traj = traj.permute(1, 0, 2, 3).contiguous().view(traj.shape[1], -1, traj.shape[3])
    save_image(traj * 0.5 + 0.5, path)
    print(f"Saved trajectory to {path}")

def store_uint128_pairs(filename, pairs):
    with open(filename, 'wb') as f:
        for a, b in pairs:
            # Convert uint128 to bytes (16 bytes each)
            a_bytes = a.to_bytes(16, byteorder='big')
            b_bytes = b.to_bytes(16, byteorder='big')
            # Write to file
            f.write(a_bytes + b_bytes)

def load_uint128_pairs(filename):
    pairs = []
    with open(filename, 'rb') as f:
        while True:
            data = f.read(32)  # 32 bytes for each pair
            if not data:
                break
            a_bytes, b_bytes = data[:16], data[16:]
            a, b = int.from_bytes(a_bytes, byteorder='big'), int.from_bytes(b_bytes, byteorder='big')
            pairs.append((a, b))
    return pairs

def store_uint128_pairs_zip(zip_path, fname, pair):
    with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zipf:
        # Convert the pair to bytes (16 bytes each)
        a_bytes = pair[0].to_bytes(16, byteorder='big')
        b_bytes = pair[1].to_bytes(16, byteorder='big')
        # Write the pair as a file in the ZIP archive, naming it with fname
        zipf.writestr(f'{fname}.bin', a_bytes + b_bytes)

def load_uint128_pairs_zip(zip_path, fname):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Construct the filename to load from the ZIP archive
        file_name = f'{fname}.bin'
        # Open the specified file and read its content
        with zipf.open(file_name, 'r') as f:
            data = f.read()
            # Split the data back into two uint128 integers
            a_bytes, b_bytes = data[:16], data[16:]
            a, b = int.from_bytes(a_bytes, byteorder='big'), int.from_bytes(b_bytes, byteorder='big')
            return (a, b)


class NoiseGenerator:
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        state_val, inc_val = self.rng.bit_generator.state['state']['state'], self.rng.bit_generator.state['state']['inc']
        state = {
            'bit_generator': 'PCG64',
            'state': {
                'state': state_val,
                'inc': inc_val
            },
            'has_uint32': False,
            'uinteger': 0
        }
        self.rng.bit_generator.state = state
    def update_state(self, state):
        # State: (state, inc) tuple
        new_state = {
            'bit_generator': 'PCG64',
            'state': {
                'state': state[0],
                'inc': state[1]
            },
            'has_uint32': False,
            'uinteger': 0
        }
        self.rng.bit_generator.state = new_state
    def get_state(self):
        # Return (state, inc) tuple
        return self.rng.bit_generator.state['state']['state'], self.rng.bit_generator.state['state']['inc']
    def _sample_noise(self, noise_shape, state = None):
        # Generate noise with the given shape. State can be given to reproduce the noise.
        # Return noise and the state used to generate the noise.
        if state is not None:
            self.update_state(state)
        else:
            state = self.get_state()
        noise = self.rng.standard_normal(noise_shape).astype(np.float32)
        return noise, state
    def sample_noise(self, noise_shape, state = None):
        # Batched version of _sample_noise with the batch size B
        B = noise_shape[0]
        if state is not None:
            assert len(state) == B, f"State length is {len(state)}, but should be {B}"
        noise_list = []
        state_list = []

        if state is None:
            for i in range(B):
                noise, state = self._sample_noise(noise_shape[1:])
                noise_list.append(noise)
                state_list.append(state)
            return np.array(noise_list), np.array(state_list)

        else:
            for i in range(B):
                noise, _ = self._sample_noise(noise_shape[1:], state[i])
                noise_list.append(noise)
            return np.array(noise_list), state