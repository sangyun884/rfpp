# 2-Rectified Flow++

This is the codebase of our paper "Improving the Training of Rectified Flows".

## Setup
Tested environment: PyTorch >= 2.0.0, Linux.

You can install the required packages by running `pip install -r requirements.txt`.

## Getting started with pre-trained 2-rectified flow++ models
We provide pre-trained 2-rectified flow++ models [here](https://drive.google.com/open?id=13cgGNkpOacb4HxlUM75ylcFOHFa0t2d1&usp=drive_fs).

### Generation
To generate 50,000 samples from the pre-trained CIFAR-10 model using 2 GPUs, run:
```
python generate.py --gpu 0,1 --dir runs/test \
--solver euler --N 5 --sampler new --num_samples 50000 --batchsize 512 \
--ckpt CKPT_PATH --config configs_unet/cifar10_ve_aug.json
```
where `CKPT_PATH` is the path to the pre-trained model checkpoint. To evaluate FID, run:
```
torchrun --standalone --nproc_per_node=1 fid.py calc --images=runs/test/samples --ref=PATH_TO_cifar10-32x32.npz --num 50000;
```

You can download inception statistics from [here](https://drive.google.com/drive/u/2/folders/1MCEAn0VdeD-lMu1Cdkm9z7q-CdzH1JDc).

### Image to image translation
Lion (291) -> Tiger (292):
```
python generate.py --gpu 0 --dir runs/test-inversion --solver euler --N 4 --N_decode 2 --batchsize 6 --ckpt imagenet-configF.pth --config configs_unet/imagenet64.json --action inversion \
--data_path imagenet-samples/n02129165/ --num_samples 6 --sampler new --label_inv 291 --label_rec 292
```

## Training

### Generating synthetic pairs
First, download the pre-trained EDM checkpoints from [here](https://drive.google.com/open?id=18dWE-LiodXdCG0RDNegySzRnyRdcwamW&usp=drive_fs).
Then, generate synthetic pairs using the pre-trained EDM models by running:
```
# Generate synthetic pairs for CIFAR-10

python generate_edm.py --gpu 0,1,2,3 --dir  runs/cifar-pair --num_samples 1000000 --batchsize 2048 --config configs_unet/cifar10_ve_aug.json --ckpt edm_cifar_ve_uncond.pth  --N 18 --ext zip

# Generate synthetic pairs for ImageNet 64x64

python generate.py --gpu 0,1,2,3 --dir  runs/imagenet-pair --num_samples 5000000 --batchsize 512 --config configs_unet/imagenet64.json --ckpt edm_imagenet64_ve_cond.pth --N 40 --ext zip
```

This will generate `samples.zip` (data), `z.zip` (noise), and `images_labels.csv` (class labels) in the specified directory.

### Training 2-rectified flow++
Run:
```
export WORLD_SIZE=1
export NODE_RANK=0
export MASTER_ADDR=localhost

# Train CIFAR-10 using Config F
python train.py --gpu 0,1,2,3 --dir runs/test-cifar --warmup_steps 5000 --learning_rate 2e-4 --batchsize 512 --iterations 800001 --config_de configs_unet/cifar10_ve_aug.json \
--ema_decay 0.9999 \
--im_dir runs/cifar-pair/samples.zip --z_dir runs/cifar-pair/z.zip \
--ckpt edm_cifar_ve_uncond.pth --a 4 --loss_type lpips-huber --lpips_divt --port 12354 \
--compile

# Train ImageNet 64x64 using Config E
python train.py --gpu 0,1,2,3,4,5,6,7 --dir runs/test-imagenet --learning_rate 1e-4 --warmup_steps 2500 \
--batchsize 512 --effective_batchsize 2048 --iterations 700001 --ema_decay 0.9999 \
--config_de HVAE/configs_unet/imagenet64.json \
--im_dir runs/imagenet-pair/samples.zip --z_dir /runs/imagenet-pair/z.zip \
--ckpt edm_imagenet64_ve_cond.pth --a 4 --loss_type lpips-huber --port 12355 \
--loss_scaling 32768 --compile

# Train AFHQ using Config F
python train.py --gpu 0,1,2,3,4,5,6,7 --dir runs/test-afhq --learning_rate 2e-4 --warmup_steps 5000 --batchsize 256 --iterations 1000001 --config_de configs_unet/afhq64_ve_aug.json \
--ema_decay 0.9999 \
--im_dir runs/afhq-pair/samples.zip --z_dir runs/afhq-pair/z.zip \
--ckpt edm_afhq64_ve_cond.pth --a 4 --loss_type lpips-huber --port 12356 --compile

```

## Citation
```
TBA
```
