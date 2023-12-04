import torch
import torch.nn as nn

eps = 1e-6


def NSS_loss(pred, gt_s, overlap_mask=torch.ones(1, 480, 960)):
    target = gt_s[:, 1]
    target_mean = target.mean([-2, -1], keepdim=True)
    target_std = target.std([-2, -1], keepdim=True)

    pred_mean = pred.mean([-2, -1], keepdim=True)
    pred_std = pred.std([-2, -1], keepdim=True)

    ref = (target - target_mean) / target_std
    pred = (pred - pred_mean) / (pred_std + eps)

    loss = (overlap_mask * (ref * target - pred * target)).sum([-2, -1]) / target.sum([-2, -1])
    return loss.mean()


def SMSE_loss(pred, gt_s, overlap_mask=torch.ones(1, 480, 960)):
    gtsal, target = gt_s[:, 0], gt_s[:, 1]
    gtsal_mean = gtsal.mean([-2, -1], keepdim=True)
    gtsal_std = gtsal.std([-2, -1], keepdim=True)
    pred_mean = pred.mean([-2, -1], keepdim=True)
    pred_std = pred.std([-2, -1], keepdim=True)

    ref = (gtsal - gtsal_mean) / gtsal_std
    pred = (pred - pred_mean) / (pred_std + eps)

    loss = ((torch.pow((ref * target - pred * target), 2) * overlap_mask).mean([-2, -1]) / target.mean([-2, -1])).mean()

    return loss


class SaliencyLoss(nn.Module):
    def __init__(self, config):
        super(SaliencyLoss, self).__init__()
        self.use_smse = config["train"]["criterion"]["use_smse"]
        self.use_infogain = config["train"]["criterion"]["use_infogain"]
        self.nss = SMSE_loss if self.use_smse else NSS_loss
        self.eps = torch.finfo(torch.float32).eps

    def KL_loss(self, pred, target, overlap_mask=torch.ones([1, 480, 960])):
        pred = pred.float()
        target = target.float()

        pred_sum = pred.sum([-2, -1], keepdim=True)
        pred = pred / (pred_sum + self.eps)

        target_sum = target.sum([-2, -1], keepdim=True)
        target = target / target_sum

        loss = (((target * torch.log(target / (pred + self.eps) + self.eps)) * overlap_mask).sum([-2, -1])).mean()
        return loss

    def CC_loss(self, pred, target, overlap_mask=torch.ones([1, 480, 960])):
        pred = pred.float()
        target = target.float()
        pred_mean = pred.mean([-2, -1], keepdim=True)
        pred_std = pred.std([-2, -1], keepdim=True)
        pred = (pred - pred_mean) / (pred_std + self.eps)

        target_mean = target.mean([-2, -1], keepdim=True)
        target_std = target.std([-2, -1], keepdim=True)
        target = (target - target_mean) / target_std
        loss = (overlap_mask * (pred * target)).sum() / (torch.sqrt((pred * pred).sum() * (target * target).sum()))
        return 1 - loss

    def forward(self, pred, gt_s, overlap_mask):
        kl = self.KL_loss(pred, gt_s[:, 0], overlap_mask)
        cc = self.CC_loss(pred, gt_s[:, 0], overlap_mask)
        nss = self.nss(pred, gt_s, overlap_mask)
        return kl, cc, nss


class VACLoss(nn.Module):
    def __init__(self):
        super(VACLoss, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.w_cc = 1.
        self.w_mse = 0.5
        self.MSE = torch.nn.MSELoss(reduction='none')

    def CC_loss(self, pred, target, overlap_mask):
        pred = pred.float()
        target = target.float()
        pred_mean = pred.mean([-2, -1], keepdim=True)
        pred_std = pred.std([-2, -1], keepdim=True)
        pred = (pred - pred_mean) / (pred_std + self.eps)

        target_mean = target.mean([-2, -1], keepdim=True)
        target_std = target.std([-2, -1], keepdim=True)
        target = (target - target_mean) / target_std
        loss = (overlap_mask * (pred * target)).sum() / (torch.sqrt((pred * pred).sum() * (target * target).sum()))

        range_max = max(1, overlap_mask.mean())
        return range_max - loss

    def forward(self, pred, gt_s, overlap_mask):
        cc = self.CC_loss(pred, gt_s, overlap_mask)
        # mse = self.MSE(pred, gt_s).mean()
        mse = self.MSELoss(pred, gt_s)
        # handle mse nan
        # if torch.isnan(mse):
        #     mse = torch.tensor(0.0).cuda()
        # if torch.isnan(cc):
        #     cc = torch.tensor(0.0).cuda()

        return cc * self.w_cc + mse * self.w_mse

    def MSELoss(self, pred, gt_s):
        # normalize
        pred = pred.float()
        target = gt_s.float()
        pred_mean = pred.mean([-2, -1], keepdim=True)
        pred_std = pred.std([-2, -1], keepdim=True)
        pred = (pred - pred_mean) / (pred_std + self.eps)

        target_mean = target.mean([-2, -1], keepdim=True)
        target_std = target.std([-2, -1], keepdim=True)
        target = (target - target_mean) / target_std

        loss = self.MSE(pred, target).mean()
        return loss