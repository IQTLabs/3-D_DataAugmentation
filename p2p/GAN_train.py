import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
from torchvision import transforms

from .utils import *

__all__ = ['GAN_train_p2p']


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


def prepare_image(img):
    view_img = img
    view_img = torch.clamp(inv_normalize(view_img), 0, 1)
    view_img = np.array(view_img)
    view_img = np.moveaxis(view_img, 0, -1)
    return view_img


def make_loss_fig(g_train, g_val, d_train, d_val, fname):
    fig = plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.plot(g_train, label='Train loss')
    plt.plot(g_val, label='Val loss')
    plt.legend()
    plt.title('Generator')
    plt.subplot(1, 2, 2)
    plt.plot(d_train, label='Train loss')
    plt.plot(d_val, label='Val loss')
    plt.legend()
    plt.title('Discriminator')
    plt.savefig(fname)
    plt.close()


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


def save_checkpoint(generator, discriminator, filename='checkpoint.pth.tar'):
    """ Saves input state dict to file
    Parameters
    ----------
    state : dict
        State dict to save. Can include parameters from model, optimizer, etc.
        as well as any other elements.
    is_best : bool
        If true will save current state dict to a second location
    filename : str
        File name for save
    Returns
    -------
    """
    state = {
        'generator': generator,
        'discriminator': discriminator
    }
    torch.save(state, filename)


def GAN_test_p2p(generator=None, discriminator=None, testloader=None,
                 g_criterion=None, d_criterion=None, fname='',
                 device_ids=[]):
    generator.to(device_ids[0])
    discriminator.to(device_ids[0])
    generator.eval()
    discriminator.eval()
    g_avg_loss = 0
    d_avg_loss = 0
    total = 0
    pbar = tqdm(total=len(testloader))
    for i_batch, batch in enumerate(testloader):
        pbar.update(1)
        frame0, frame1, pose1 = batch
        frame0, frame1, pose1 = frame0.to(
            device_ids[0]), frame1.to(device_ids[0]), pose1.to(device_ids[0])
        dis_true_label = torch.full(
            (frame0.shape[0], 1), 0.95).to(device_ids[0])
        dis_fake_label = torch.full(
            (frame0.shape[0], 1), 0.05).to(device_ids[0])
        with torch.no_grad():
            bs = frame0.shape[0]
            preds = generator(
                torch.cat([frame0, pose1], dim=1).to(device_ids[0]))
            pred_real = discriminator(frame1)
            pred_fake = discriminator(preds)
            d_loss = 0.5*(d_criterion(pred_real, dis_true_label) +
                          d_criterion(pred_fake, dis_fake_label))
            g_loss = g_criterion(preds, frame1) + \
                d_criterion(pred_fake, dis_true_label)
            d_avg_loss += bs*d_loss.item()
            g_avg_loss += bs*g_loss.item()
            total += frame0.shape[0]
        if i_batch == 0:
            make_fig(frame0[0].cpu(), frame1[0].cpu(), preds[0].cpu(), fname)
    pbar.close()
    return g_avg_loss/total, d_avg_loss/total


def GAN_train_p2p(generator=None, discriminator=None, trainloader=None,
                  testloader=None, g_opt=None, g_sched=None, g_criterion=None,
                  d_opt=None, d_sched=None, d_criterion=None, n_epochs=0,
                  e_saves=10, save_path='',  device_ids=[],
                  verbose=False):
    """ Training routing for deep fake detector
    Parameters
   ----------
    model : torch.Module
        Deep fake detector model
    dataloader : torch.utils.data.DataLoader
        Training dataset
    optim : torch.optim
        Optimizer for pytorch model
    scheduler : torch.optim.lr_scheduler
        Optional learning rate scheduler for the optimizer
    criterion : torch.nn.Module
        Objective function for optimization
    losses : list
        List to hold the lossses over each mini-batch
    averages : list
        List to hold the average loss over each epoch
    n_epochs : int
        Number of epochs for training
    device_ids[0] : str
        Device_Ids[0] to run training procedure
    verbose : bool
        Verbose switch to print losses at each mini-batch
    """
    generator = generator.to(device_ids[0])
    discriminator = discriminator.to(device_ids[0])
    g_meter = AverageMeter()
    d_meter = AverageMeter()
    g_avg_train = []
    g_avg_val = []
    d_avg_train = []
    d_avg_val = []
    chpt_path = '{}/models'.format(save_path)
    log_path = '{}/plots'.format(save_path)
    Path(chpt_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose is False:
        pbar = tqdm(total=len(trainloader))
    for epoch in range(n_epochs):
        generator.train()
        discriminator.train()
        for i_batch, batch in enumerate(trainloader):
            #
            frame0, frame1, pose1 = batch
            frame0, frame1, pose1 = frame0.to(
                device_ids[0]), frame1.to(device_ids[0]), pose1.to(device_ids[0])
            g_opt.zero_grad()
            d_opt.zero_grad()
            #
            dis_true_label = torch.full(
                (frame0.shape[0], 1), 0.95).to(device_ids[0])
            dis_fake_label = torch.full(
                (frame0.shape[0], 1), 0.05).to(device_ids[0])
            #
            preds = generator(
                torch.cat([frame0, pose1], axis=1).to(device_ids[0]))
            pred_real = discriminator(frame1)
            pred_fake = discriminator(preds.detach())
            d_loss = 0.5*(d_criterion(pred_real, dis_true_label) +
                          d_criterion(pred_fake, dis_fake_label))
            d_loss.backward()
            d_opt.step()
            d_meter.update(d_loss.item(), frame0.shape[0])
            #
            pred_fake = discriminator(preds)
            g_loss = g_criterion(preds, frame1) + \
                d_criterion(pred_fake, dis_true_label)
            g_loss.backward()
            g_opt.step()
            g_meter.update(g_loss.item(), frame0.shape[0])
            pbar.update(1)
        if verbose is False:
            pbar.refresh()
            pbar.reset()
        g_val_loss, d_val_loss = GAN_test_p2p(generator, discriminator, testloader,
                                              g_criterion, d_criterion,
                                              '{}/sammple_{}.jpg'.format(
                                                  log_path, epoch),
                                              device_ids)
        if (epoch+1) % e_saves == 0:
            save_checkpoint(
                generator.state_dict(), discriminator.state_dict(),
                '{}/checkpoint_{}.pth.tar'.format(chpt_path, epoch))
        g_avg_train.append(g_meter.avg)
        g_avg_val.append(g_val_loss)
        d_avg_train.append(d_meter.avg)
        d_avg_val.append(d_val_loss)

        make_loss_fig(g_avg_train, g_avg_val, d_avg_train,
                      d_avg_val, '{}/losses.jpg'.format(log_path))
        if g_sched is not None:
            g_sched.step(g_meter.avg)
        if d_sched is not None:
            d_sched.step(d_meter.avg)
        g_meter.reset()
        d_meter.reset()
    pbar.close()
