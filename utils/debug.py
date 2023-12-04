import matplotlib.pyplot as plt
import torch


def imshow(s, title=''):
    plt.imshow(s)
    plt.title(title)
    plt.show()


def denorm(s):
    """
    s: tensor of shape (C, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return s * std + mean

def print_stats(tensor):
    print(f"mean: {tensor.mean()}")
    print(f"std: {tensor.std()}")
    print(f"min: {tensor.min()}")
    print(f"max: {tensor.max()}")

# i = 0
# imshow(pred_erp[i], f"pred {i}")
# imshow(y_erp[i, 0], f"gt {i}")
