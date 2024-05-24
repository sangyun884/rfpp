# https://github.com/NVlabs/edm/blob/main/generate.py

import torch
import numpy as np
import os
from torchvision.utils import save_image, make_grid
from utils import parse_config, save_traj, NoiseGenerator, store_uint128_pairs_zip
import argparse
from tqdm import tqdm
from network_edm import SongUNet, DhariwalUNet, EDMPrecond
from torch.nn import DataParallel
import json
import zipfile
from PIL import Image
import io
from torchvision.transforms import functional as TF
import csv

DEBUG = os.getenv('DEBUG') == '1'
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')
    parser.add_argument('--N', type=int, default = 18, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--config', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--rho', type=float, default=7, help='rho')
    parser.add_argument('--euler', action='store_true', help='Use Euler method')
    parser.add_argument('--ext', type=str, default = "zip", help='Extension of the generated images. png / zip')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    


    arg = parser.parse_args()
    return arg

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, heun=True
):
    _round_sigma = net.round_sigma if hasattr(net, 'round_sigma') else net.module.round_sigma
    _sigma_min = net.sigma_min if hasattr(net, 'sigma_min') else net.module.sigma_min
    _sigma_max = net.sigma_max if hasattr(net, 'sigma_max') else net.module.sigma_max
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, _sigma_min)
    sigma_max = min(sigma_max, _sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([_round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    t_steps_print = t_steps.cpu().numpy()
    # # Round
    t_steps_print = np.round(t_steps_print, 4)
    print(t_steps_print)

    # Main sampling loop.
    traj_x0 = []
    traj = []
    x_next = latents.to(torch.float64) * t_steps[0]
    traj.append(x_next.cpu())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        t_cur = torch.ones(latents.shape[0], device='cuda') * t_cur
        t_next = torch.ones(latents.shape[0], device='cuda') * t_next
        t_cur = t_cur.to(torch.float64).view(-1, 1, 1, 1)
        t_next = t_next.to(torch.float64).view(-1, 1, 1, 1)

        # Increase noise temporarily.
        # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # t_hat = _round_sigma(t_cur + gamma * t_cur)
        t_hat = t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur


        # Apply 2nd order correction.
        if i < num_steps - 1 and heun:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        traj_x0.append(denoised.cpu())
        traj.append(x_next.cpu())

    return x_next, traj_x0, traj

@torch.no_grad()
def main(arg):

    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config is not None
    config = parse_config(arg.config)
    arg.res = config['img_resolution']
    arg.input_nc = config['in_channels']
    arg.label_dim = config['label_dim']


    if not os.path.exists(os.path.join(arg.dir, "samples")):
        os.makedirs(os.path.join(arg.dir, "samples"))
    if not os.path.exists(os.path.join(arg.dir, "zs")):
        os.makedirs(os.path.join(arg.dir, "zs"))
    if not os.path.exists(os.path.join(arg.dir, "trajs")):
        os.makedirs(os.path.join(arg.dir, "trajs"))
    if not os.path.exists(os.path.join(arg.dir, "data")):
        os.makedirs(os.path.join(arg.dir, "data"))
    # Create tmp directory for torch.compile
    if not os.path.exists(os.path.join(arg.dir, "tmp")):
        os.makedirs(os.path.join(arg.dir, "tmp"))
    os.environ['TMPDIR'] = os.path.join(arg.dir, "tmp")
    
    if arg.num_samples > 60000:
        num_subfolders = arg.num_samples // 60000 + 1
        for i in range(num_subfolders):
            if not os.path.exists(os.path.join(arg.dir, f"zs/{i}")):
                os.makedirs(os.path.join(arg.dir, f"zs/{i}"))
            if not os.path.exists(os.path.join(arg.dir, f"samples/{i}")):
                os.makedirs(os.path.join(arg.dir, f"samples/{i}"))
    else:
        num_subfolders = 0
    arg.num_subfolders = num_subfolders
    


    if config['unet_type'] == 'adm':
        model_class = DhariwalUNet
    elif config['unet_type'] == 'songunet':
        model_class = SongUNet
    unet = model_class(**config)

    # Print the number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in unet.parameters())
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of the reverse parameters: {pytorch_total_params}M")
    # Save the configuration of unet to a json file
    config_dict = unet.config
    config_dict['num_params'] = pytorch_total_params
    with open(os.path.join(arg.dir, 'config_unet.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)

    denoiser = EDMPrecond(unet)

    device_ids = arg.gpu.split(',')
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
    else:
        device = torch.device(f"cuda")
        print(f"Using GPU {arg.gpu}!")
    

    if arg.ckpt is not None:
        denoiser.load_state_dict(torch.load(arg.ckpt, map_location = "cpu"))
    else:
        raise NotImplementedError("Model ckpt should be provided.")
    if len(device_ids) > 1:
        denoiser = DataParallel(denoiser)
    denoiser = denoiser.to(device).eval()
    if arg.compile:
        assert len(device_ids) == 1, f"Compile is not supported for DataParallel."
        denoiser = torch.compile(denoiser)


    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)

    noise_gen = NoiseGenerator(arg.seed)
    ######################### Generation #########################
    i = 0
    epoch = arg.num_samples // arg.batchsize + 1
    zip_path_z = os.path.join(arg.dir, "z.zip")
    zip_batch_z = []
    zip_batchsize = 10000


    if arg.ext == "zip":
        zip_path_samples = os.path.join(arg.dir, "samples.zip")
        zip_batch_im = []

    for ep in tqdm(range(epoch)):
        z, z_state = noise_gen.sample_noise((arg.batchsize, arg.input_nc, arg.res, arg.res))
        z = torch.tensor(z).to(device)
        # z = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
        if arg.label_dim > 0:
            # Generate random label
            label_onehot = torch.eye(arg.label_dim, device = device)[torch.randint(0, arg.label_dim, (z.shape[0],), device = device)]
        else:
            label_onehot = None
        x0, traj, _ = edm_sampler(denoiser, z, label_onehot, num_steps=arg.N, rho = arg.rho, heun = not arg.euler)
        x0 = (x0 * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        # z_file = zipfile.ZipFile(os.path.join(arg.dir, "z.zip"), 'a')
        with open(os.path.join(arg.dir, "images_labels.csv"), 'a') as f:
            writer = csv.writer(f)
            for idx in range(len(x0)):
                if arg.num_subfolders > 0:
                    subfolder_idx = i // 60000
                    path_img = os.path.join(arg.dir, "samples", f"{subfolder_idx}", f"{i:05d}.png")
                    path_z = os.path.join(f"{subfolder_idx}", f"{i:05d}")
                else:
                    path_img = os.path.join(arg.dir, "samples", f"{i:05d}.png")
                    path_z = os.path.join(f"{i:05d}")
                img = x0[idx]
                if img.shape[2] == 1:
                    img = Image.fromarray(img[:, :, 0], 'L')
                else:
                    img = Image.fromarray(img, 'RGB')
                if arg.ext == "png":
                    img.save(path_img)
                elif arg.ext == "zip":
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    zip_batch_im.append((path_z + ".png", img_byte_arr))
                    if len(zip_batch_im) >= zip_batchsize:
                        with zipfile.ZipFile(zip_path_samples, 'a') as z_file:
                            for path, img_byte_arr in zip_batch_im:
                                z_file.writestr(path, img_byte_arr)
                        zip_batch_im = []
                if label_onehot is not None:
                    # Save argmax of the label
                    label = torch.argmax(label_onehot[idx]).item()
                    writer.writerow([path_z + ".png", label])

                # Save z as npy file
                a_bytes = z_state[idx][0].to_bytes(16, byteorder='big')
                b_bytes = z_state[idx][1].to_bytes(16, byteorder='big')
                zip_batch_z.append((path_z, (a_bytes, b_bytes)))
                if len(zip_batch_z) >= zip_batchsize:
                    with zipfile.ZipFile(zip_path_z, 'a') as z_file:
                        for path, (a_bytes, b_bytes) in zip_batch_z:
                            z_file.writestr(f'{path}.bin', a_bytes + b_bytes)
                    zip_batch_z = []


                i+=1
                if i >= arg.num_samples:
                    break
    
        
        if arg.save_traj:
            save_traj(traj, os.path.join(arg.dir, "trajs", f"{ep:05d}_traj.png"))
        if i >= arg.num_samples:
            break
    if arg.ext == "zip" and len(zip_batch_im) > 0:
        with zipfile.ZipFile(zip_path_samples, 'a') as z_file:
            for path, img_byte_arr in zip_batch_im:
                z_file.writestr(path, img_byte_arr)
    if len(zip_batch_z) > 0:
        with zipfile.ZipFile(zip_path_z, 'a') as z_file:
            for path, (a_bytes, b_bytes) in zip_batch_z:
                z_file.writestr(f'{path}.bin', a_bytes + b_bytes)

if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)