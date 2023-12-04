import argparse
import numpy as np
import os
import random
import torch
import wandb
import yaml

from dotmap import DotMap
from utils.trainer import Trainer

seed = 3536
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)


def main_video(config, dataset_config):
    phase = config.phase.name

    trainer = Trainer(config, dataset_config)

    if phase == 'train':
        trainer.train()
    elif phase == 'val':
        trainer.eval(-1)
    else:
        raise NotImplementedError("Invalid phase!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='Path to the config file'
    )

    parser.add_argument(
        '--dataset', '-d', type=str, required=True,
        help='Path to the dataset config file'
    )

    parser.add_argument(
        '--wandb', type=str, choices=['online', 'offline', 'disabled'], default='disabled',
    )

    parser.add_argument('--gpus', help='comma-seperated list of GPU(s) to use. Default=0', type=lambda s: [int(item) for item in s.split(',')])

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model_config = args.config
    dataset_config = args.dataset

    gpus = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])

    config = DotMap(yaml.safe_load(open(model_config, 'r')))
    dataset_config = DotMap(yaml.safe_load(open(dataset_config, 'r')))

    os.environ['WANDB_MODE'] = args.wandb
    wandb.init(project=config.wandb.project, name=f"Config {config.Config}-{config.wandb.name}", config=config)

    main_video(config, dataset_config)
