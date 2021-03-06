import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from .config import args
from .models import Vgg19

__all__ = ['AverageMeter', 'createOptim', 'make_fig', 'VGGLoss']


COLORS = [[255, 0, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [0, 255, 255],
          [0, 170, 255], [0, 0, 255], [170, 0, 255], [255, 0, 255]]

pose_joints = {'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
               'l_brow': [17, 18, 19, 20, 21],
               'r_brow': [22, 23, 24, 25, 26],
               'bridge': [27, 28, 29, 30],
               'nostrils': [31, 32, 33, 34, 35],
               'l_eye': [36, 37, 38, 39, 40, 41, 36],
               'r_eye': [42, 43, 44, 45, 46, 47, 42],
               'outer_mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48],
               'inner_mouth': [60, 61, 62, 63, 64, 65, 66, 67, 60]}


inv_normalize = transforms.Normalize(
    mean=[-args['norm_mean'][x]/args['norm_std'][x] for x in range(3)],
    std=[1./args['norm_std'][x] for x in range(3)]
)


def prepare_image(img):
    """ Prepres image for plotting
    Parameters
    ----------
    img : torch.tensor
        Image to prepare for plotting size=(3, H, W)
    Returns
    -------
    view_img : np.array
        Array with image ready for pyplot size (H, W, 3)
    """
    view_img = img
    view_img = torch.clamp(inv_normalize(view_img), 0, 1)
    view_img = np.array(view_img)
    view_img = np.moveaxis(view_img, 0, -1)
    return view_img


def make_fig(in_frame, target, pred, fname):
    """ Make figure to visualize performance
    Parameters
    ----------
    in_frame : torch.tensor
        Input frame tensor
    target : torch.tensor
        Target (real) frame
    pred : torch.tensor
        Predicted target frame
    fname : str
        Output file name
    Results
    -------
    """
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
    """  VGG or Perceptual loss. Matches target and predition features
    extracted from pretrained (ImageNet) VGG19
    """

    def __init__(self, device, use_mse=False):
        """ Loss module initializaiton
        Parameters
        ----------
        device : torch.device
            Device to run trianing/inference
        Returns
        -------
        """
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # undoing generator norm
        self.inv_mean = torch.tensor(
            [-args['norm_mean'][x]/args['norm_std'][x] for x in range(3)]).to(device)
        self.inv_mean = self.inv_mean.view(-1, 1, 1)

        self.inv_std = torch.tensor(
            [1./args['norm_std'][x] for x in range(3)]).to(device)
        self.inv_std = self.inv_std.view(-1, 1, 1)
        # imagenet norm
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.mean = self.mean.view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.std = self.std.view(-1, 1, 1)
        self.use_mse = use_mse

    def renormalize(self, tensor):
        """ Generator to ImageNet normalization
        Parameters
        ----------
        tensor : torch.tensor
            Image tensor to renormalize to accomodate ImageNet norm   
        Returns
         -------
        new_tensor : torch.tensor
            Renormalized image tensor
        """
        new_tensor = (tensor-self.inv_mean)/self.inv_std
        return (new_tensor-self.mean)/self.std

    def forward(self, x, y):
        """ Forward pass
        Parameters
        ----------
        x : torch tensor
            Predicted image tensor
        y : tensor
            Target image tensor
        Returns
         -------
        loss : torch.tensor
            Batch loss
        """
        loss = 0
        if self.use_mse:
            loss += torch.nn.MSELoss()(x, y)
        x = self.renormalize(x)
        y = self.renormalize(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
