import torch
import numpy as np
import torch.nn as nn
import tensorboardX
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness, save_traj
from dataset import DatasetWithLatentCond
import argparse
from tqdm import tqdm
import json 
from EMA import EMA
from network_edm import SongUNet, EDMPrecondVel, DhariwalUNet
from time import time
from t_dist import ExponentialPDF, sample_t
import torch.nn.functional as F
import matplotlib.pyplot as plt

# DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch_utils.misc import InfiniteSampler
from fp16_utils import DynamicLossScaler
from piq import LPIPS
from generate import sample_ode_generative

torch.manual_seed(0)

def ddp_setup(local_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        port: Port number to use for initialization
    """
    print(f"Master address is {os.environ['MASTER_ADDR']}")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = master_addr
    rank = local_rank + num_gpus_per_node * node_rank
    world_size = num_nodes * num_gpus_per_node
    # Windows
    # init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # Linux
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Initialized on port {port}")

def get_args():
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu num')
    parser.add_argument('--im_dir', type=str, help='Image dir')
    parser.add_argument('--z_dir', type=str, help='zs dir')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--tmpdir', type=str, help='Temporary directory', default=None)
    parser.add_argument('--iterations', type=int, default = 1000000, help='Number of iterations')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')
    parser.add_argument('--effective_batchsize', type=int, default = None, help='Effective batch size. If None, same as batchsize. If larger than batchsize, gradient accumulation is used')
    parser.add_argument('--learning_rate', type=float, default = 3e-5, help='Learning rate')
    parser.add_argument('--resume', type=str, default = None, help='Training state path')
    parser.add_argument('--ckpt', type=str, default = None, help='Model ckpt path')
    parser.add_argument('--N', type=int, default = 32, help='Number of sampling steps')
    parser.add_argument('--no_ema', action='store_true', help='use EMA or not')
    parser.add_argument('--ema_after_steps', type=int, default = 1, help='Apply EMA after steps')
    parser.add_argument('--ema_decay', type=float, default = 0.9999, help='EMA decay rate')
    parser.add_argument('--save_iter', type=int, default = 50000, help='Save iteration')
    parser.add_argument('--optimizer', type=str, default = 'adam', help='adam / adamw')
    parser.add_argument('--warmup_steps', type=int, default = 0, help='Learning rate warmup')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--t_dist', type=str, default = 'uniform', help='weighting, [uniform, exponential-inc, exponential-dec]')
    parser.add_argument('--a', type=float, default = 2, help='alpha for exponential distribution')
    parser.add_argument('--loss_type', type=str, default = 'l2', help='loss type, [l2, lpips, huber]')
    parser.add_argument('--port', type=int, default = 12354, help='Port number')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')


    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument('--subset', type=int, default = None, help='Subset of the dataset')

    parser.add_argument('--loss_scaling', type=float, default = 1, help='Loss scaling factor')

    parser.add_argument('--lpips_divt', action='store_true', help='Divide lpips by t')

    arg = parser.parse_args()

    arg.use_ema = not arg.no_ema
    return arg


def train_rectified_flow(rank, arg, model, optimizer, data_loader, iterations, device, start_iter, warmup_steps, dir, learning_rate, 
                         ema_after_steps, use_ema, sampling_steps, world_size, save_iter):
    if rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=dir)
    # use tqdm if rank == 0

    gradient_accumulation_steps = arg.effective_batchsize // arg.batchsize
    if rank == 0:
        log = f"gradient_accumulation_steps: {gradient_accumulation_steps}"
        print(log)
        with open(os.path.join(dir, "log.txt"), "a") as f:
            f.write(log + "\n")
    i_effective = start_iter
    cnt = 0 # Count the number of backward() calls
    iterations_effective = (iterations - start_iter) * gradient_accumulation_steps # Total number of backward() calls
    iterations_effective += 1000 # Since we sometimes skip the update, safely add some extra iterations

    noise_fixed = None # For visualization
    label_fixed_onehot = None # For visualization

    tqdm_ = tqdm if rank == 0 else lambda x: x

    # Define loss function
    if arg.loss_type == 'lpips':
        loss_lpips = LPIPS(replace_pooling=True, reduction="none")
        if arg.compile:
            loss_lpips = torch.compile(loss_lpips)

        def loss_func(x, y):
            return loss_lpips(x * 0.5 + 0.5, y * 0.5 + 0.5) 
    elif arg.loss_type == 'l2-squared':
        def loss_func(x, y):
            return torch.mean((x - y)**2, dim = (1, 2, 3))
    elif arg.loss_type == 'l2':
        def loss_func(x, y):
            return torch.sqrt(torch.mean((x - y)**2, dim = (1, 2, 3)))
    elif arg.loss_type == 'huber':
        def loss_func(x, y):
            data_dim = x.shape[1] * x.shape[2] * x.shape[3]
            huber_c = 0.00054 * data_dim
            loss = torch.sum((x - y)**2, dim = (1, 2, 3))
            loss = torch.sqrt(loss + huber_c**2) - huber_c
            return loss / data_dim
    elif arg.loss_type == 'lpips-huber':
        loss_lpips = LPIPS(replace_pooling=True, reduction="none")
        if arg.compile:
            loss_lpips = torch.compile(loss_lpips)

        def loss_func_huber(x, y):
            data_dim = x.shape[1] * x.shape[2] * x.shape[3]
            huber_c = 0.00054 * data_dim
            loss = torch.sum((x - y)**2, dim = (1, 2, 3))
            loss = torch.sqrt(loss + huber_c**2) - huber_c
            return loss / data_dim
        def loss_func_lpips(x, y):
            return loss_lpips(x * 0.5 + 0.5, y * 0.5 + 0.5)
    else:
        raise NotImplementedError(f"Loss type {arg.loss_type} not implemented")


        
    # Initialize timstep distribution
    exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')

    # Save histogram
    if rank == 0:
        t_samples = sample_t(exponential_distribution, 50000, arg.a).numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(t_samples, bins=50, density=True)
        plt.savefig(os.path.join(dir, f'exponential_samples.png'), dpi=300)
        plt.close()

    train_iter = iter(data_loader)
    optimizer.zero_grad()

    loss_scaler = DynamicLossScaler(init_scale=arg.loss_scaling, scale_window = 10000)

    for cnt in tqdm_(range(iterations_effective)):
        if use_ema and i_effective > ema_after_steps:
            optimizer.ema_start()
        # Learning rate warmup
        if i_effective < warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * np.minimum(i_effective / warmup_steps, 1)
        
        # Load data
        x, z, c, _ = next(train_iter)
        if arg.label_dim > 0:
            c = c.to(device)
        else:
            c = None
        x = x.to(device)
        z = z.to(device)

        # Initialize noise_fixed and label_fixed_onehot for visualization
        if noise_fixed is None:
            noise_fixed = torch.randn((32, *z.shape[1:]), device = device) * (1-1e-5)
            if arg.label_dim > 0:
                label_fixed = np.array([c_ % arg.label_dim for c_ in range(noise_fixed.shape[0])], dtype=np.int64)
                label_fixed_onehot = torch.zeros(noise_fixed.shape[0], arg.label_dim, device=device)
                label_fixed_onehot[np.arange(noise_fixed.shape[0]), label_fixed] = 1
        
        # Sample t, zt
        t = sample_t(exponential_distribution, x.shape[0], arg.a).to(device) # (batchsize,)
        zt = (1-t).view(-1, 1, 1, 1) * x + t.view(-1, 1, 1, 1) * z
        target = z - x            
        
        # Forward pass
        pred = model(zt, t, c)

        # Predicted x for LPIPS loss
        pred_x = zt - pred * t.view(-1, 1, 1, 1)
        pred_x_up = F.interpolate(pred_x, size=224, mode="bilinear")
        x_up = F.interpolate(x, size=224, mode="bilinear")

        # Compute loss
        loss_dict = {}
        if arg.loss_type == 'lpips':
            if arg.lpips_divt:
                loss = loss_func(pred_x_up, x_up) / t.squeeze()
            else:
                loss = loss_func(pred_x_up, x_up)
            loss_dict['lpips'] = loss.mean().item()
        elif arg.loss_type == 'lpips-huber':
            loss_huber = loss_func_huber(pred, target)
            loss_lp = loss_func_lpips(pred_x_up, x_up)

            if arg.lpips_divt:
                loss = (1-(t).squeeze()) * loss_huber + loss_lp / t.squeeze()
            else:
                loss = (1-(t).squeeze()) * loss_huber + loss_lp
            loss_dict['lpips'] = loss_lp.mean().item()
            loss_dict['huber'] = loss_huber.mean().item()
        else:
            loss = loss_func(pred, target)
            loss_dict[arg.loss_type] = loss.mean().item()


        loss = loss.mean()

        # Loss scaling for mixed precision training
        if arg.loss_scaling == 1:
            loss_scale = 1
        else:
            loss_scale = loss_scaler.loss_scale

        (loss * loss_scale / gradient_accumulation_steps).backward()
        cnt += 1

        has_overflow = loss_scaler.has_overflow(model.parameters())
        loss_scaler.update_scale(has_overflow)


        if cnt % gradient_accumulation_steps == 0:
            if not has_overflow:
                if loss_scale != 1:
                    for param in model.parameters():
                        param.grad.data *= 1 / loss_scale
                optimizer.step()
                optimizer.zero_grad()
                i_effective += 1
            else:
                log = f"Overflow at iteration {i_effective}"
                print(log)
                with open(os.path.join(dir, 'log.txt'), 'a') as f:
                    f.write(log + "\n")
                optimizer.zero_grad()

        else:
            if has_overflow:
                log = f"Overflow at iteration {i_effective}"
                print(log)
                with open(os.path.join(dir, 'log.txt'), 'a') as f:
                    f.write(log + "\n")
                optimizer.zero_grad()
            continue # Skip logging, visualization, and saving

        ########### Logging, visualization, and saving ###########
        
        if i_effective % 100 == 1 and rank == 0:
            log = f"Iteration {i_effective}: lr {optimizer.param_groups[0]['lr']} "
            for key in loss_dict:
                log += f"{key} {loss_dict[key]:.8f} "
            log += f"loss_scale {loss_scale:.8f}"
            log += "\n"
            print(log)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], i_effective)
            writer.add_scalar("loss", loss.item(), i_effective)
            writer.add_scalar("loss_scale", loss_scale, i_effective)
            for key in loss_dict:
                writer.add_scalar(key, loss_dict[key], i_effective)
            # Log to .txt file
            with open(os.path.join(dir, 'log.txt'), 'a') as f:
                f.write(log)
        if i_effective % 1000 == 5 and rank == 0:
            # model.eval() # Doesn't work with torch.compile
            if use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)

            with torch.no_grad():                
                traj_uncond, traj_uncond_x0 = sample_ode_generative(model, z1=noise_fixed, N=sampling_steps, label = label_fixed_onehot)
                traj_uncond_N4, traj_uncond_x0_N4 = sample_ode_generative(model, z1=noise_fixed, N=4,  label = label_fixed_onehot)
                uncond_straightness = straightness(traj_uncond)

                print(f"Uncond straightness: {uncond_straightness.item()}")
                writer.add_scalar("uncond_straightness", uncond_straightness.item(), i_effective)
                # Log to .txt file
                with open(os.path.join(dir, 'log.txt'), 'a') as f:
                    f.write(f"Uncond straightness: {uncond_straightness.item():.8f} \n")

                save_traj(traj_uncond, os.path.join(dir, f"traj_uncond_{i_effective}.jpg"))
                save_traj(traj_uncond_x0, os.path.join(dir, f"traj_uncond_x0_{i_effective}.jpg"))
                save_traj(traj_uncond_N4, os.path.join(dir, f"traj_uncond_N4_{i_effective}.jpg"))
                save_traj(traj_uncond_x0_N4, os.path.join(dir, f"traj_uncond_x0_N4_{i_effective}.jpg"))
            if use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            # model.train()

        if i_effective % save_iter == 0 and rank == 0:
            if use_ema:
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                torch.save(model.module.state_dict(), os.path.join(dir, f"flow_model_{i_effective}_ema.pth"))
                optimizer.swap_parameters_with_ema(store_params_in_ema=True)
            else:
                torch.save(model.module.state_dict(), os.path.join(dir, f"flow_model_{i_effective}.pth"))

            # Save training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = model.module.state_dict()
            d['iter'] = i_effective
            # save
            torch.save(d, os.path.join(dir, f"training_state_{i_effective}.pth"))  
        if i_effective % 5000 == 0 and rank == 0 and i_effective > 0:
            # Save the latest training state
            d = {}
            d['optimizer_state_dict'] = optimizer.state_dict()
            d['model_state_dict'] = model.module.state_dict()
            d['iter'] = i_effective
            # save
            torch.save(d, os.path.join(dir, f"training_state_latest.pth"))  

    return

def get_loader(arg, world_size, rank):
    train_dataset = DatasetWithLatentCond(arg.im_dir, arg.z_dir, input_nc = arg.input_nc, label_dim = arg.label_dim)
    if arg.subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(arg.subset))
    # Print len
    if rank == 0:
        print(f"len(train_dataset) = {len(train_dataset)}")

    data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=arg.batchsize,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=arg.num_workers,
                                            pin_memory=True,
                                            sampler = InfiniteSampler(train_dataset, num_replicas=world_size, rank=rank)
                                            )

   
    return data_loader, arg.res, arg.input_nc

def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
def main(local_rank: int, num_nodes: int, num_gpus_per_node: int, node_rank: int, master_addr: str, arg):
    port = arg.port

    rank = local_rank + num_gpus_per_node * node_rank
    world_size = num_nodes * num_gpus_per_node

    ddp_setup(local_rank, num_nodes, num_gpus_per_node, node_rank, master_addr, port)


    device = torch.device(f"cuda:{local_rank}")
    assert arg.config_de is not None
    config_de = parse_config(arg.config_de)

    arg.res = config_de['img_resolution']
    arg.input_nc = config_de['in_channels']
    arg.label_dim = config_de['label_dim']
    

    data_loader, res, input_nc = get_loader(arg, world_size, rank)

    if config_de['unet_type'] == 'songunet':
        model_class = SongUNet
    elif config_de['unet_type'] == 'adm':
        model_class = DhariwalUNet
    flow_model = model_class(**config_de)

    if rank == 0:
        # Print the number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
        # Convert to M
        pytorch_total_params = pytorch_total_params / 1000000
        print(f"Total number of the reverse parameters: {pytorch_total_params}M")
        # Save the configuration of flow_model to a json file
        config_dict = flow_model.config
        config_dict['num_params'] = pytorch_total_params
        with open(os.path.join(arg.dir, 'config_flow_model.json'), 'w') as f:
            json.dump(config_dict, f, indent = 4)
    
    # EDM
    if 'use_fp16' not in config_de:
        config_de['use_fp16'] = False
    flow_model = EDMPrecondVel(flow_model, use_fp16 = config_de['use_fp16'])


    # Load training state in arg.training_state
    if arg.resume is not None:
        training_state = torch.load(arg.resume, map_location = 'cpu')
        start_iter = training_state['iter']
        flow_model.load_state_dict(training_state['model_state_dict'])
    else:
        start_iter = 0
    if arg.ckpt is not None:
        flow_model.load_state_dict(torch.load(arg.ckpt, map_location = 'cpu'))

    flow_model = flow_model.to(device)

    
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=arg.learning_rate, betas = (0.9, 0.999), eps=1e-8)

    if arg.use_ema:
        optimizer = EMA(optimizer, ema_decay=arg.ema_decay)

    if arg.resume is not None:
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        print(f"Loaded training state from {arg.resume} at iter {start_iter}")
        del training_state


    
    # DDP
    flow_model = DDP(flow_model, device_ids=[local_rank])
    if arg.compile:
        flow_model = torch.compile(flow_model)# mode="reduce-overhead" raises an error
    

    train_rectified_flow(rank = rank, arg = arg, model = flow_model, optimizer = optimizer,
                        data_loader = data_loader, iterations = arg.iterations, device = device, start_iter = start_iter,
                        warmup_steps = arg.warmup_steps, dir = arg.dir, learning_rate = arg.learning_rate,
                        use_ema = arg.use_ema, ema_after_steps = arg.ema_after_steps, sampling_steps = arg.N, world_size=world_size,
                        save_iter = arg.save_iter)
    destroy_process_group()

if __name__ == "__main__":
    arg = get_args()

    device_ids = arg.gpu.split(',')
    device_ids = [int(i) for i in device_ids]

    # Process environment variables
    num_nodes = int(os.environ['WORLD_SIZE'])
    num_gpus_per_node = len(device_ids)
    node_rank = int(os.environ['NODE_RANK'])
    master_addr = os.environ['MASTER_ADDR']

    if node_rank == 0:
        if not os.path.exists(arg.dir):
            os.makedirs(arg.dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    if arg.tmpdir is None:
        arg.tmpdir = os.path.join(arg.dir, "tmp")

    # Create tmp directory for torch.compile
    if not os.path.exists(arg.tmpdir):
        os.makedirs(arg.tmpdir)
    os.environ['TMPDIR'] = arg.tmpdir

    # world_size = len(device_ids)
    if node_rank == 0:
        with open(os.path.join(arg.dir, "config.json"), "w") as json_file:
            json.dump(vars(arg), json_file, indent = 4)
    # Gradient accumulation
    if arg.effective_batchsize is None:
        arg.effective_batchsize = arg.batchsize
    else:
        assert arg.effective_batchsize >= arg.batchsize
        assert arg.effective_batchsize % arg.batchsize == 0

    log = f"num_nodes: {num_nodes}, num_gpus_per_node: {num_gpus_per_node}, node_rank: {node_rank}, master_addr: {master_addr}, batchsize: {arg.batchsize}, effective_batchsize: {arg.effective_batchsize}"
    print(log)
    if node_rank == 0:
        with open(os.path.join(arg.dir, "log.txt"), "a") as f:
            f.write(log + "\n")

    # DDP
    arg.batchsize = arg.batchsize // num_nodes // num_gpus_per_node
    arg.effective_batchsize = arg.effective_batchsize // num_nodes // num_gpus_per_node
    try:
       mp.spawn(main, args=(num_nodes, num_gpus_per_node, node_rank, master_addr, arg), nprocs=num_gpus_per_node)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        destroy_process_group()
        exit(0)