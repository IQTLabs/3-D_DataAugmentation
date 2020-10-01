import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from .config import args
from .models import Vgg19

__all__ = ['AverageMeter', 'createOptim', 'make_fig', 'VGGLoss']


inv_normalize = transforms.Normalize(
    mean=[-args['norm_mean'][x]/args['norm_std'][x] for x in range(3)],
    std=[1./args['norm_std'][x] for x in range(3)]
)


def prepare_image(img):
    view_img = img
    view_img = torch.clamp(inv_normalize(view_img), 0, 1)
    view_img = np.array(view_img)
    view_img = np.moveaxis(view_img, 0, -1)
    return view_img


def make_fig(in_frame, target, pred, fname):
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    img = prepare_image(in_frame.cpu())
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    img = prepare_image(target.cpu())
    plt.imshow(img)
    plt.subplot(1, 3, 3)
    img = prepare_image(pred.cpu())
    plt.imshow(img)
    plt.savefig(fname)
    plt.close()


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        """ Initialize objects and reset for safety
        Parameters
        ----------
        Returns
        -------
        """
        self.reset()

    def reset(self):
        """ Resets the meter values if being re-used
        Parameters
        ----------
        Returns
        -------
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update meter values give current value and batchsize
        Parameters
        ----------
        val : float
            Value fo metric being tracked
        n : int
            Batch size
        Returns
        -------
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def createOptim(parameters, lr=0.001, betas=(0.5, 0.999), weight_decay=0,
                factor=0.2, patience=5, threshold=1e-03,  eps=1e-08):
    """ Creates optimizer and associated learning rate scheduler for a model
    Paramaters
    ----------
    parameters : torch parameters
        Pytorch network parameters for associated optimzer and scheduler
    lr : float
        Learning rate for optimizer
    betas : 2-tuple(floats)
        Betas for optimizer
    weight_decay : float
        Weight decay for optimizer regularization
    factor : float
        Factor by which to reduce learning rate on Plateau
    patience : int
        Patience for learning rate scheduler
    Returns
    -------
    optimizer : torch.optim
        optimizer for model
    scheduler : ReduceLROnPlateau
        scheduler for optimizer
    """
    optimizer = optim.Adam(parameters, lr=lr, betas=(
        0.5, 0.999), weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=patience,
        threshold=threshold, eps=eps, verbose=True)
    return optimizer, scheduler


class VGGLoss(torch.nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
