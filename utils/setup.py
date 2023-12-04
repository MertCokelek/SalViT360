import os
import torch

from model.SalViT360_VAC import SalViT360_VAC
from model.SalViT360 import SalViT360_VSTA
from utils.saliency_losses import SaliencyLoss


def get_criterion(config, model=None):
    criterion = SaliencyLoss(config)
    loss_weights = config['train']['criterion']['weights']
    return criterion, loss_weights


def get_optimizer_and_scheduler(model, config):
    
    # Optimizer
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }

    optim_algorithm = config['train']['optim_algorithm']
    optim = config['train']['optim'][optim_algorithm]
    optimizer = optimizers[optim_algorithm](filter(lambda p: p.requires_grad, model.parameters()), **optim)
    print("[Optimizer]:", optim_algorithm)

    # Scheduler
    schedulers = {
        'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    }
    
    sched_algorithm = config['train']['sched_algorithm']
    sched = config['train']['sched'][sched_algorithm]
    scheduler = schedulers[sched_algorithm](optimizer, **sched)

    print("[Scheduler]:", sched_algorithm)

    return optimizer, scheduler


def get_model(config, erp_size=(960, 1920)):

    def load_pretrained(model, config):
        ckpt = config.network.resume
        if ckpt:
            assert os.path.exists(ckpt), "Checkpoint does not exist!"
            print('Loading state dict from:', os.path.basename(ckpt))
            pt_model = torch.load(os.path.abspath(os.path.expanduser(ckpt)), map_location='cpu')['model']
            
            ckpt = {k.replace("module.", ""): v for k, v in pt_model.items()}
            missing, unexpected = model.load_state_dict(ckpt, strict=False)

            if len(missing) > 0:
                print(f"Missing keys: {missing}")
                
            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected}")
        else:
            print("No checkpoint provided. Training from scratch.")
        return model
    
    if config.train.criterion.vac:
        model = SalViT360_VAC(config, erp_size)
    else:
        model = SalViT360_VSTA(config, erp_size)

    model = load_pretrained(model, config)
    model = torch.nn.DataParallel(model).cuda()
    return model


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable params: {trainable}/{total}")


def save_checkpoint(checkpoint, epoch, save_dir, config_id):
    out_path = os.path.join(os.path.expanduser(save_dir), f'Config {config_id}')
    os.makedirs(out_path, exist_ok=True)
    out_path = f"{out_path}/Epoch_{epoch}.pt"
    torch.save(checkpoint, out_path)
    print('Checkpoint saved at', out_path)


def set_to_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False

    model.requires_grad_(False)