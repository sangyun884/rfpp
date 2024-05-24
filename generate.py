# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from flows import RectifiedFlow
import torch.nn as nn
# import tensorboardX
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness, parse_config, save_traj
import argparse
from tqdm import tqdm
from network_edm import SongUNet, DhariwalUNet, EDMPrecondVel
from torch.nn import DataParallel
import json
import matplotlib.pyplot as plt
from PIL import Image
import glob

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')

    parser.add_argument('--N', type=int, default = 20, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--save_z', action='store_true', help='Save zs for distillation')    
    parser.add_argument('--save_data', action='store_true', help='Save data')    
    parser.add_argument('--solver', type=str, default = 'euler', help='ODE solvers')
    parser.add_argument('--config', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--im_dir', type=str, help='Image dir')

    parser.add_argument('--action', type=str, default='sample', help='sample or interpolate')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--t_steps', type=str, default = None, help='t_steps, e.g. 1,0.75,0.5,0.25')
    parser.add_argument('--sampler', type=str, default = 'default', help='default / new')

    # Inversion
    parser.add_argument('--data_path', type=str, default = None, help='Image path for inversion')
    parser.add_argument('--label_inv', type=int, help='Label for inversion')
    parser.add_argument('--label_rec', type=int, help='Label for reconstruction')    
    parser.add_argument('--N_decode', type=int, default = 5, help='Number of decoding sampling steps')



    arg = parser.parse_args()
    return arg


def slerp(val, low, high):
    # val: interpolation parameter, low & high: two end points.
    assert low.shape == high.shape, f"low.shape: {low.shape}, high.shape: {high.shape}"
    low_clone = low.clone()
    low_shape = low.shape
    
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)) + 1e-8
    so = torch.sin(omega)
    interpolated = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return interpolated

@torch.no_grad()
def sample_ode_generative(model, z1, N, use_tqdm=True, solver = 'euler', label = None, inversion = False, time_schedule = None, sampler='default'):
    assert solver in ['euler', 'heun']
    assert len(z1.shape) == 4
    if inversion:
        assert sampler == 'default'
    tq = tqdm if use_tqdm else lambda x: x

    if solver == 'heun' and N % 2 == 0:
        raise ValueError("N must be odd when using Heun's method.")
    if solver == 'heun':
        N = (N + 1) // 2
    traj = [] # to store the trajectory
    x0hat_list = []
    z1 = z1.detach()
    z = z1.clone()
    batchsize = z.shape[0]
    if time_schedule is not None:
        time_schedule = time_schedule + [0]
        sigma_schedule = [t_ / (1-t_ + 1e-6) for t_ in time_schedule]
        print(f"sigma_schedule: {sigma_schedule}")
    else:
        t_func = lambda i: i / N
        if inversion:
            time_schedule = [t_func(i) for i in range(0,N)] + [1]
            time_schedule[0] = 1e-3
        else:
            time_schedule = [t_func(i) for i in reversed(range(1,N+1))] + [0]
            time_schedule[0] = 1-1e-5

    cnt = 0
    print(f"Time schedule: {time_schedule}")

    

    
    config = model.module.config if hasattr(model, 'module') else model.config
    if config["label_dim"] > 0 and label is None:
        label = torch.randint(0, config["label_dim"], (batchsize,)).to(z1.device)
        label = F.one_hot(label, num_classes=config["label_dim"]).type(torch.float32)

    traj.append(z.detach().clone())
    for i in tq(range(len(time_schedule[:-1]))):
        t = torch.ones((batchsize), device=z1.device) * time_schedule[i]
        t_next = torch.ones((batchsize), device=z1.device) * time_schedule[i+1]
        dt = t_next[0] - t[0]

        vt = model(z, t, label)
        if inversion:
            x0hat = z + vt * (1-t).view(-1,1,1,1) # z-prediction
        else:
            x0hat = z - vt * t.view(-1,1,1,1) # x-prediction

        # Heun's correction
        if solver == 'heun' and cnt < N - 1:
            if sampler == 'default' or inversion:
                z_next = z.detach().clone() + vt * dt
            elif sampler == 'new':
                z_next = (1 - t_next.view(-1,1,1,1)) * x0hat + t_next.view(-1,1,1,1) * z1
            else:
                raise NotImplementedError(f"Sampler {sampler} not implemented.")

            vt_next = model(z_next, t_next, label)
            vt = (vt + vt_next) / 2

            if inversion:
                x0hat = z_next + vt_next * (1-t_next).view(-1,1,1,1) # z-prediction
            else:
                x0hat = z_next - vt_next * t_next.view(-1,1,1,1) # x-prediction
    
        x0hat_list.append(x0hat)
        
        if inversion:
            x0hat = z + vt * (1-t).view(-1,1,1,1)
        else:
            x0hat = z - vt * t.view(-1,1,1,1)
        
        if sampler == 'default' or inversion:
            z = z.detach().clone() + vt * dt
        elif sampler == 'new':
            z = (1 - t_next.view(-1,1,1,1)) * x0hat + t_next.view(-1,1,1,1) * z1
        else:
            raise NotImplementedError(f"Sampler {sampler} not implemented.")
        cnt += 1
        traj.append(z.detach().clone())

    return traj, x0hat_list
    
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
    if not os.path.exists(os.path.join(arg.dir, 'tmp')):
        os.makedirs(os.path.join(arg.dir, 'tmp'))
    os.environ['TMPDIR'] = os.path.join(arg.dir, 'tmp')
    
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
    # Pass the arguments in the config file to the model
    flow_model = model_class(**config)

    device_ids = arg.gpu.split(',')
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
    else:
        device = torch.device(f"cuda")
        print(f"Using GPU {arg.gpu}!")
    # Print the number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of parameters: {pytorch_total_params}M")

    if 'use_fp16' not in config:
        config['use_fp16'] = False
    flow_model = EDMPrecondVel(flow_model, use_fp16 = config['use_fp16'])

    if arg.ckpt is not None:
        flow_model.load_state_dict(torch.load(arg.ckpt, map_location = "cpu"))
    else:
        raise NotImplementedError("Model ckpt should be provided.")
    if len(device_ids) > 1:
        flow_model = DataParallel(flow_model)
    flow_model = flow_model.to(device).eval()

    if arg.compile:
        flow_model = torch.compile(flow_model, mode="reduce-overhead", fullgraph=True)








    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)

    if arg.action == 'sample':
       sample(arg, flow_model, device)
    elif arg.action == 'inversion':
        inversion(arg, flow_model, device)
    elif arg.action == 'sample_from_npy':
        sample_from_npy(arg, flow_model, device)
    else:
        raise NotImplementedError(f"{arg.action} is not implemented")

@torch.no_grad()
def sample(arg, model, device):
    i = 0
    epoch = arg.num_samples // arg.batchsize + 1
    x0_list = []
    straightness_list = []
    nfes = []
    for ep in tqdm(range(epoch)):
        z = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
      
        # Compute the norm of z
        if arg.label_dim > 0:
            # Generate random label
            label_onehot = torch.eye(arg.label_dim, device = device)[torch.randint(0, arg.label_dim, (z.shape[0],), device = device)]
        else:
            label_onehot = None
        if arg.solver in ['euler', 'heun']:
            if arg.t_steps is not None:
                t_steps = [float(t) for t in arg.t_steps.split(",")]
                t_steps[0] = 1-1e-5 # max t value
            else:
                t_steps = None
            z = z * (1-1e-5)
            traj_uncond, traj_uncond_x0 = sample_ode_generative(model, z1=z, N=arg.N, use_tqdm = False, solver = arg.solver, label = label_onehot, time_schedule = t_steps, sampler = arg.sampler)
            x0 = traj_uncond[-1]
            uncond_straightness = straightness(traj_uncond, mean = False)
            straightness_list.append(uncond_straightness)
        else:
            raise NotImplementedError(f"{arg.solver} is not implemented")

        if arg.save_traj:
            save_traj(traj_uncond, os.path.join(arg.dir, "trajs", f"{ep:05d}_traj.png"))
            save_traj(traj_uncond_x0, os.path.join(arg.dir, "trajs", f"{ep:05d}_traj_x0.png"))
            
        for idx in range(len(x0)):
            if arg.num_subfolders > 0:
                subfolder_idx = i // 60000
                path_img = os.path.join(arg.dir, "samples", f"{subfolder_idx}", f"{i:05d}.png")
                path_z = os.path.join(arg.dir, "zs", f"{subfolder_idx}", f"{i:05d}.npy")
            else:
                path_img = os.path.join(arg.dir, "samples", f"{i:05d}.png")
                path_z = os.path.join(arg.dir, "zs", f"{i:05d}.npy")
            save_image(x0[idx] * 0.5 + 0.5, path_img)
            # Save z as npy file
            if arg.save_z:
                np.save(path_z, z[idx].cpu().numpy())
            if arg.save_data:
                save_image(x[idx] * 0.5 + 0.5, os.path.join(arg.dir, "data", f"{i:05d}.png"))
            i+=1
            if i >= arg.num_samples:
                break
    straightness_list = torch.stack(straightness_list).view(-1).cpu().numpy()
    straightness_mean = np.mean(straightness_list).item()
    straightness_std = np.std(straightness_list).item()
    print(f"straightness.shape: {straightness_list.shape}")
    print(f"straightness_mean: {straightness_mean}, straightness_std: {straightness_std}")
    nfes_mean = np.mean(nfes) if len(nfes) > 0 else arg.N
    print(f"nfes_mean: {nfes_mean}")
    result_dict = {"straightness_mean": straightness_mean, "straightness_std": straightness_std, "nfes_mean": nfes_mean}
    with open(os.path.join(arg.dir, 'result_sampling.json'), 'w') as f:
        json.dump(result_dict, f, indent = 4)
    # Plot the distribution of straightness (sum to one)
    plt.hist(straightness_list, bins = 20)

    plt.savefig(os.path.join(arg.dir, "straightness.png"))


@torch.no_grad()
def inversion(arg, model, device):
    im_names = glob.glob(os.path.join(arg.data_path, "*.jpg")) + glob.glob(os.path.join(arg.data_path, "*.png")) + glob.glob(os.path.join(arg.data_path, "*.JPEG"))
    # Name only
    im_names = [os.path.basename(im_name) for im_name in im_names]
    im_names.sort()
    im_names = im_names[:arg.num_samples]
    print(f"im_names: {im_names}")
    num_epoch = len(im_names) // arg.batchsize + 1
    if arg.label_dim > 0:
        assert arg.label_inv is not None
        assert arg.label_rec is not None
        print(f"label_inv: {arg.label_inv}, label_rec: {arg.label_rec}")
        label_onehot_inv = torch.zeros(1, arg.label_dim, device = device)
        label_onehot_inv[0, arg.label_inv] = 1
        label_onehot_inv = label_onehot_inv.repeat(arg.batchsize, 1)

        label_onehot_interp = torch.zeros(1, arg.label_dim, device = device)
        label_onehot_interp[0, arg.label_rec] = 1
        label_onehot_interp = label_onehot_interp.repeat(10, 1)

        label_onehot_rec = torch.zeros(1, arg.label_dim, device = device)
        label_onehot_rec[0, arg.label_rec] = 1
        label_onehot_rec = label_onehot_rec.repeat(arg.batchsize, 1)
    else:
        label_onehot_inv = None
        label_onehot_rec = None
        label_onehot_interp = None
    for epoch in tqdm(range(num_epoch)):
        print(f"epoch: {epoch}")
        im_list = []
        for idx in range(arg.batchsize):
            img = Image.open(os.path.join(arg.data_path, im_names[epoch * arg.batchsize + idx]))
            img = img.resize((arg.res, arg.res), Image.LANCZOS)
            # to tensor
            img = transforms.ToTensor()(img) * 2 - 1
            im_list.append(img)
        img = torch.stack(im_list, dim = 0).to(device) 
        img_in = (1-1e-3) * img + 1e-3 * torch.randn_like(img)
        
        z_list, z_traj = sample_ode_generative(model, z1=img_in, N=arg.N, use_tqdm = True, solver = arg.solver, inversion = True, label = label_onehot_inv)
        z = z_list[-1].to(device)


        # Interpolation
        lamb = torch.linspace(0, 1, 10, device = device)
        z0 = z[1].unsqueeze(0)
        z1 = z[2].unsqueeze(0)
        z_interp = slerp(lamb, z0.view(z0.shape[0], -1), z1.view(z1.shape[0], -1)).view(-1, *z0.shape[1:])

        N_decode = arg.N_decode
        x_recon_list_interp, traj_recon_interp = sample_ode_generative(model, z1=z_interp* (1-1e-5), N=N_decode, use_tqdm = False, solver = arg.solver, sampler = arg.sampler, label = label_onehot_interp)
        x_recon_interp = x_recon_list_interp[-1]

        # Recon
        x_recon_list, traj_recon = sample_ode_generative(model, z1=z * (1-1e-5), N=N_decode, use_tqdm = False, solver = arg.solver, sampler = arg.sampler, label = label_onehot_rec)
        x_recon = x_recon_list[-1]

        print(f"MSE: {torch.mean((img - x_recon) ** 2)}")
     
        # Save as npy
        for latent, name in zip(z.detach().cpu().numpy(), im_names[epoch * arg.batchsize: epoch * arg.batchsize + arg.batchsize]):
            # Remove file extension, e.g., sample_0001.jpeg -> sample_0001, use split
            name = name.split(".")[0]
            np.save(os.path.join(arg.dir, "zs", f"{name}.npy"), latent)
        print(f"Saved {epoch * arg.batchsize} to {epoch * arg.batchsize + arg.batchsize} / {len(im_names)}")

        for img, name in zip(img, im_names[epoch * arg.batchsize: epoch * arg.batchsize + arg.batchsize]):
            save_image(img*0.5 + 0.5, os.path.join(arg.dir, "samples", f"{name}"))
        for x_recon, name in zip(x_recon, im_names[epoch * arg.batchsize: epoch * arg.batchsize + arg.batchsize]):
            save_image(x_recon*0.5 + 0.5, os.path.join(arg.dir, "samples", f"recon_{name}"))
        # Save z as .png
        for z_, name in zip(z, im_names[epoch * arg.batchsize: epoch * arg.batchsize + arg.batchsize]):
            save_image(z_*0.5 + 0.5, os.path.join(arg.dir, "samples", f"noise_{name}"))
        # Save x_recon_interp
        save_image(x_recon_interp * 0.5 + 0.5, os.path.join(arg.dir, "samples", f"{epoch:05d}_recon_interp.png"), nrow = x_recon_interp.shape[0], padding = 0)

        # Saver traj_recon_interp
        save_traj(traj_recon_interp, os.path.join(arg.dir, "trajs", f"{epoch:05d}_traj_interp.png"))
        # Save z_list
        save_traj(z_list, os.path.join(arg.dir, "trajs", f"{epoch:05d}_traj_z.png"))
        # Save z_traj
        save_traj(z_traj, os.path.join(arg.dir, "trajs", f"{epoch:05d}_traj.png"))
        save_traj(traj_recon, os.path.join(arg.dir, "trajs", f"{epoch:05d}_traj_recon.png"))

        # Save z as .npz file
        np.savez(os.path.join(arg.dir, "zs", f"{epoch:05d}_z.npz"), z = z.detach().cpu().numpy())
        print(f"z.shape: {z.shape}")
        

if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)