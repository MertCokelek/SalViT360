import os
import wandb
import random
import torch
import numpy as np

from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from dataset.videoclip_dataset import VideoClipDataset

from utils.setup import count_parameters
from utils.metrics import inner_worker
from utils.setup import get_model, get_criterion, get_optimizer_and_scheduler
from utils.debug import imshow, denorm, print_stats

TRAIN = "TRAIN"
EVAL = "EVAL"


class Trainer:
    def __init__(self, config, dataset_config):
        model = get_model(config, (dataset_config.h, dataset_config.w))
        self.VAC = config.train.criterion.vac
        if self.VAC:
            print('Using Viewport Augmentation Consistency (VAC) Loss')

        # Criterion, Optimizer, Scheduler
        criterion, loss_weights = get_criterion(config)
        self.optimizer, self.scheduler = get_optimizer_and_scheduler(model, config)

        self.use_amp = config.network.use_amp
        print('\nUsing AMP:', self.use_amp)

        self.model = model
        self.config = config
        self.dataset_config = dataset_config
        self.criterion = criterion

        self.w_kl = loss_weights['kl']
        self.w_cc = loss_weights['cc']
        self.w_nss = loss_weights['nss']
        
        self.w_sup = loss_weights['w_sup'] if 'w_sup' in loss_weights.keys() else 1. 
        self.w_vac = loss_weights['w_vac'] if 'w_vac' in loss_weights.keys() else 1.
        print('Sup', self.w_sup, 'VAC', self.w_vac)

        self.epoch_st = config.train.epoch.start
        self.epoch_en = self.epoch_st + config.train.epoch.n_epochs

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        videoclip_root = dataset_config["videoclip"]["root"]
        self.n_workers = dataset_config["num_workers"]

        # Train set
        self.train_clip_paths = sorted(glob(f"{videoclip_root}/train/*.mp4"))
        random.Random(4).shuffle(self.train_clip_paths)
        self.train_BS = config.train["train_bs"]

        # Validation set
        val_clip_paths = sorted(glob(f"{videoclip_root}/val/*.mp4"))
        random.Random(4).shuffle(val_clip_paths)
        n_clips_val = int(len(val_clip_paths) * config.train.val_subset)
        self.val_clip_paths = val_clip_paths[:n_clips_val]
        self.val_BS = config.train["val_bs"]


        self.n_jobs = dataset_config['n_jobs'] if 'n_jobs' in dataset_config.keys() else 4
        self.save_ckpt_dir = self.config['save']
        self.config_id = self.config.Config

        self.overlap_mask = torch.ones(1).cuda()  # VAC Mask initialized as ones. Will be updated later.

        self.model.module.freeze_resnet()
        # self.model.module.freeze_saliency()

        print("Video Backbone, ", end='')
        count_parameters(self.model)

        self.hflip = config.train.use_data_augmentation


    def train(self):
        for epoch in range(self.epoch_st, self.epoch_en):
            i = 0
            self.set_to_train()
            pbar = tqdm(range(len(self.train_clip_paths) // self.train_BS), desc=f'Train Epoch {epoch}')
            for _ in pbar:
                clip_paths = self.train_clip_paths[i: i + self.train_BS]
                i += self.train_BS
                
                train_dl = DataLoader(
                    VideoClipDataset(self.config, self.dataset_config, clip_paths),
                    num_workers=self.n_workers
                )
                
                for batch in train_dl:
                    x_erp, y_erp = batch
                    flip = random.randint(0, 1) if self.hflip else 0
                    if flip:
                        x_erp = torch.flip(x_erp, [-1])
                        y_erp = torch.flip(y_erp, [-1])
                    
                    x_erp = x_erp.cuda(non_blocking=True).squeeze(0)
                    y_erp = y_erp.cuda(non_blocking=True).squeeze(0)

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        if self.VAC:
                            pred_erp, loss_vac = self.model(x_erp, return_p0=False)
                            pred_erp = pred_erp.squeeze(1)
                            loss_vac = loss_vac.mean()
                        else:
                            pred_erp = self.model(x_erp).squeeze(1)

                        loss_kl, loss_cc, loss_nss = self.criterion(pred_erp, y_erp, self.overlap_mask)
                        loss_nss = loss_nss * self.w_nss
                        loss = self.w_kl * loss_kl + self.w_cc * loss_cc + self.w_nss * loss_nss

                        postfix = {
                            'loss': loss.item(),
                            'loss_kl': loss_kl.item(),
                            'loss_cc': loss_cc.item(),
                            'loss_nss': loss_nss.item()
                        } 
                        if self.VAC:
                            loss = self.w_sup * loss + self.w_vac * loss_vac
                            # loss += loss_vac
                            if torch.isnan(loss_vac):
                                wandb.alert(title="NaN Loss", text="Loss is NaN!")
                                exit(-1)

                            postfix['loss_vac'] = loss_vac.item()

                        pbar.set_postfix(postfix)

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.log(loss, loss_kl, loss_cc, loss_nss)

                self.scheduler.step()

            self.save_checkpoint(epoch)
            self.eval(epoch)


    @torch.no_grad()
    def eval(self, epoch):

        KL, CC, SIM, AUC, NSS = [], [], [], [], []
        self.model.eval()

        iters = len(self.val_clip_paths) // self.val_BS
        
        pbar = tqdm(range(iters), desc=f'Eval Epoch {epoch}')

        i = 0
        for iter in pbar:
            clip_paths = self.val_clip_paths[i: i + self.val_BS]
            i += self.val_BS
            val_dl = DataLoader(
                VideoClipDataset(self.config, self.dataset_config, clip_paths),
                num_workers=self.n_workers
            )
            auc, kl, cc, nss, sim = [], [], [], [], []

            for batch in val_dl:
                x_erp, y_erp = batch
                x_erp = x_erp.cuda(non_blocking=True).squeeze(0)
                y_erp = y_erp.data.cpu().squeeze(0).numpy()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.VAC:
                        pred_erp, _ = self.model(x_erp, return_p0=True)
                        pred_erp = pred_erp.squeeze(1)
                    else:
                        pred_erp = self.model(x_erp).squeeze(1)
                    pred_erp = pred_erp.data.cpu().numpy()

                metric_list = Parallel(n_jobs=self.n_jobs)(
                    delayed(inner_worker)(pred_erp[i], y_erp[i]) for i in range(x_erp.shape[0]))

                auc.append(np.nanmean([x["AUC-J"] for x in metric_list]))
                nss.append(np.nanmean([x["NSS"] for x in metric_list]))
                kl.append(np.nanmean([x["KL"] for x in metric_list]))
                cc.append(np.nanmean([x["CC"] for x in metric_list]))
                sim.append(np.nanmean([x["SIM"] for x in metric_list]))

            AUC.append(np.mean(auc))
            KL.append(np.mean(kl))
            NSS.append(np.mean(nss))
            CC.append(np.mean(cc))
            SIM.append(np.mean(sim))

            print(f"Validation - {iter}/{iters}\n\t")
         
            print(f"\tAUC: {AUC[-1]:.4f}\n\tNSS: {NSS[-1]:.4f}\n\tKL: {KL[-1]:.4f}\n\tCC: {CC[-1]:.4f}\n\tSIM: {SIM[-1]:.4f}")

        AUC = np.mean(AUC)
        KL = np.mean(KL)
        NSS = np.mean(NSS)
        CC = np.mean(CC)
        SIM = np.mean(SIM)

        line = f"Validation - Total\n\t"
        f"\tKL: {KL:.4f}\n\tCC: {CC:.4f}\n\tNSS: {NSS:.4f}\n\tSIM: {SIM:.4f}\n\tAUC-J: {AUC:.4f}"

        print(line)

        wandb.log({
            f"AUC-J (val-total)": AUC,
            f"NSS (val-total)": NSS,
            f"KL (val-total)": KL,
            f"CC (val-total)": CC,
            f"SIM (val-total)": SIM}
        )

    def set_to_train(self):
        self.model.train()
        self.model.module.feature_extractor.eval()

    def log(self, ltotal, loss_kl, loss_cc, loss_nss):
        wandb.log({
            f"Loss (train)": ltotal.item(),
            f"Loss KL (train)": loss_kl.item(),
            f"Loss CC (train)": loss_cc.item(),
            f"Loss NSS (train)": loss_nss.item(),
            }
        )

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
        }

        out_path = os.path.join(os.path.expanduser(self.save_ckpt_dir), f'Config {self.config_id}')
        os.makedirs(out_path, exist_ok=True)
        out_path = f"{out_path}/Epoch_{epoch}.pt"
        torch.save(checkpoint, out_path)
        print('Checkpoint saved at', out_path)

