import sys;

import os
import argparse

import torch
from omegaconf import OmegaConf

from exps.diffusion import diffusion
from exps.first_stage import first_stage

from utils import set_random_seed


parser = argparse.ArgumentParser()
parser.add_argument('--base_config', type=str, default='configs/base.yaml')


def main():
    """ Additional args ends here. """
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    args.exp         = base_config.base.exp
    args.seed  = base_config.base.seed
    args.id        = base_config.base.id
    args.data  = base_config.base.data
    args.batch_size       = base_config.base.batch_size
    args.ds   = base_config.base.ds
    args.pretrain_config  = base_config.base.pretrain_config
    args.diffusion_config = base_config.base.diffusion_config
    args.first_stage_folder = base_config.base.first_stage_folder
    args.first_model = base_config.base.first_model
    args.scale_lr = base_config.base.scale_lr

    """ FIX THE RANDOMNESS """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.n_gpus = torch.cuda.device_count()

    # init and save configs
    
    """ RUN THE EXP """
    if args.exp == 'ddpm':
        config = OmegaConf.load(args.diffusion_config)
        first_stage_config = OmegaConf.load(args.pretrain_config)

        args.unetconfig = config.model.params.unet_config
        args.lr         = config.model.base_learning_rate
        args.scheduler  = config.model.params.scheduler_config
        args.res        = first_stage_config.model.params.ddconfig.resolution
        args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
        args.skip       = first_stage_config.model.params.ddconfig.skip
        args.ddconfig   = first_stage_config.model.params.ddconfig
        args.embed_dim  = first_stage_config.model.params.embed_dim
        args.ddpmconfig = config.model.params
        args.cond_model = config.model.cond_model

        if args.n_gpus == 1:
            diffusion(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=diffusion, args=(args, ), nprocs=args.n_gpus)
    
    elif args.exp == 'first_stage':
       
        config = OmegaConf.load(args.pretrain_config)
        args.ddconfig   = config.model.params.ddconfig
        args.embed_dim  = config.model.params.embed_dim
        args.lossconfig = config.model.params.lossconfig
        args.lr         = config.model.base_learning_rate
        args.res        = config.model.params.ddconfig.resolution
        args.timesteps  = config.model.params.ddconfig.timesteps
        args.skip       = config.model.params.ddconfig.skip
        args.resume     = config.model.resume
        args.amp        = config.model.amp
        if args.n_gpus == 1:
            first_stage(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=first_stage, args=(args, ), nprocs=args.n_gpus)
        
    else:
        raise ValueError("Unknown experiment.")

if __name__ == '__main__':
    main()