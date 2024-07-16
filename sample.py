import os
import torch
import argparse
import numpy as np
from math import ceil
from unet import Unet
from dataloader_cifar import transback
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
import ipdb
from cifar import CIFAR10
from dataloader_cifar import load_data, transback
import matplotlib.pyplot as plt
import textwrap

# only use one GPU.
@torch.no_grad()
def sample(params:argparse.Namespace):
    
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    init_process_group(backend="nccl")
    # get local rank for each process
    local_rank = get_rank()
    # set device
    device = torch.device("cuda", local_rank)
    # load models

    if not os.path.exists(params.samdir):
        print(f'create directory {params.samdir}')
        os.makedirs(params.samdir)
    # load the dataset.
    data_sample, _, _ = load_data(params.genbatch, params.numworkers, params.noise_type, params.noise_path, params.is_human)
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv=params.useconv,
                droprate = params.droprate,
                # num_heads = params.numheads,
                dtype=params.dtype
            ).to(device)
    checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{params.epoch}_checkpoint.pt'), map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    # net.load_state_dict(torch.load(os.path.join(params.moddir, f'2nd_ckpt_{params.epoch}_diffusion.pt')))
    cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim).to(device)
    cemblayer.load_state_dict(checkpoint['cemblayer'])
    # cemblayer.load_state_dict(torch.load(os.path.join(params.moddir, f'2nd_ckpt_{params.epoch}_cemblayer.pt')))
    # settings for diffusion model
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    diffusion.model.eval()
    cemblayer.eval()
    cnt = torch.cuda.device_count()
    each_device_batch = params.genbatch // cnt

    lab_all = data_sample.train_noisy_labels_diff
    indices_diff_labels = data_sample.indices # indices of different labels in the dataset.
    img_all = data_sample.train_data
    num_loop = ceil(len(lab_all) / params.genbatch)
    all_samples = []
    all_labels = []

    for i in range(num_loop):
        if i == num_loop - 1:
            lab = torch.tensor(lab_all[i * params.genbatch : ]).to(device)
        else:
            lab = torch.tensor(lab_all[i * params.genbatch : (i+1) * params.genbatch]).to(device)
    
    # used for testing the visualization of generated samples.
    # for i in range(2):
    #     lab = torch.tensor(lab_all[i * params.genbatch : (i+1) * params.genbatch]).to(device)
        indices_gt = indices_diff_labels[i * params.genbatch : (i+1) * params.genbatch]
        img_gts = img_all[indices_gt]
        img_gts = img_gts / 255.0
        lab = lab.transpose(0, 1)
        cemb = cemblayer(lab)
        genshape = (each_device_batch, 3, 32, 32)
        if params.ddim:
            generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
        else:
            generated = diffusion.sample(genshape, cemb = cemb)
        # transform samples into images
        samples = transback(generated)  # (params.genbatch, 3, 32, 32) 
        all_samples.append(samples.cpu())
        all_labels.append(lab.cpu())

        if params.save_vis and i < num_loop - 1:
            save_visulization_samples(samples, params, lab, img_gts, i)
    # save the generated samples.
    all_samples = torch.cat(all_samples, dim = 0)  
    all_labels = torch.concat(all_labels, dim = 1)
    if params.save_data_pt:
        torch.save({'samples': all_samples, 'random_label1': all_labels[0], 'random_label2': all_labels[1], 'random_label3': all_labels[2]}, 
                   os.path.join(params.data_pt_path, 'gen_samples_and_lab_disagree_x1.pt'))


def save_visulization_samples(samples, params, lab, img_gts, iteration):
    fig, axes = plt.subplots(nrows=params.clsnum, ncols=params.genbatch // params.clsnum, figsize=(12, params.genbatch // params.clsnum))
    axes = axes.flatten()

    label = lab.transpose(0, 1)
    samples = [sample.cpu().numpy().transpose(1, 2, 0) for sample in samples]
    
    for idx, (sample, img_gt, lab_) in enumerate(zip(samples, img_gts, label.cpu().numpy())):
        classes_lab_ = '\n'.join([cifar10_classes_dict[i] for i in lab_]) 
        ax = axes[idx]
        ax.imshow(np.concatenate([sample, img_gt], axis=0))
        ax.set_title(f'{classes_lab_}')
        ax.axis('off')

    plt.tight_layout()
    img_name = os.path.join(params.samdir, f'sample_{params.epoch}_pict_{params.w}_{str(iteration)}.png')
    plt.savefig(img_name)
    plt.close()


def main():
    # several hyperparameters for models
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--genbatch',type=int,default=32, help='batch size for sampling process')
    parser.add_argument('--numworkers',type=int,default=4,help='num of workers for dataloader')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--w',type=float,default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=1.0,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for loading models')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    # parser.add_argument('--device',default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),help='devices for training Unet model')
    parser.add_argument('--moddir',type=str,default='/home/wenjie/projects/classifier-free-diffusion-guidance-Pytorch/model_cifar10n',help='model addresses')
    parser.add_argument('--samdir',type=str,default='samples_gt_cifar10n_disagreement',help='sample addresses')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0,help='dropout rate for model')
    parser.add_argument('--clsnum',type=int,default=4, help='num of label classes') # in our case, this is just used for visualization.
    parser.add_argument('--fid',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='generate samples used for quantative evaluation')
    parser.add_argument('--genum',type=int,default=5600,help='num of generated samples')
    parser.add_argument('--num_steps',type=int,default=50, help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    # Cifar10 setting.
    parser.add_argument('--noise_type', type=str, choices='[clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3, multi_rater]', 
                        default='multi_rater', help='type of noise')
    parser.add_argument('--noise_path', type=str, 
                        default='/home/wenjie/projects/classifier-free-diffusion-guidance-Pytorch/cifar-10-batches-py/CIFAR-10_human.pt', 
                        help='path to noise file')
    parser.add_argument('--is_human', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=True, help='whether to use human noise')
    parser.add_argument('--save_vis', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False, help='whether to save visualization samples')
    parser.add_argument('--save_data_pt', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=True, help='whether to save generated samples and labels')
    parser.add_argument('--data_pt_path', type=str, default='/home/wenjie/projects/classifier-free-diffusion-guidance-Pytorch/cifar-10-batches-py', help='path to save generated samples and labels')
    args = parser.parse_args()
    sample(args)

if __name__ == '__main__':
    cifar10_classes_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    main()
