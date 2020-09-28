import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch

from .utils import *

__all__ = ['train_p2p']


def save_checkpoint(generator, filename='checkpoint.pth.tar'):
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
        'generator': generator
    }
    torch.save(state, filename)


def test_p2p(generator=None, testloader=None, criterion=None, fname='',
             device_ids=[]):
    generator.to(device_ids[0])
    generator.eval()
    avg_loss = 0
    total = 0
    pbar = tqdm(total=len(testloader))
    for i_batch, batch in enumerate(testloader):
        pbar.update(1)
        frame0, frame1, pose1 = batch
        frame0, frame1, pose1 = frame0.to(
            device_ids[0]), frame1.to(device_ids[0]), pose1.to(device_ids[0])
        with torch.no_grad():
            bs = frame0.shape[0]
            preds = generator(
                torch.cat([frame0, pose1], dim=1).to(device_ids[0]))
            loss = criterion(preds, frame1)
            avg_loss += bs*loss.item()
            total += frame0.shape[0]
        if i_batch == 0:
            make_fig(frame0[0].cpu(), frame1[0].cpu(), preds[0].cpu(), fname)
    pbar.close()
    return avg_loss/total


def train_p2p(generator=None, faceid=None, trainloader=None, testloader=None,
              optim=None, scheduler=None, criterion=None, n_epochs=0,
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
    if faceid is not None:
        faceid = faceid.to(device_ids[0])
        faceid.eval()
        face_criterion = torch.nn.CosineEmbeddingLoss()
    meter = AverageMeter()
    avg_train = []
    avg_val = []
    chpt_path = '{}/models'.format(save_path)
    log_path = '{}/plots'.format(save_path)
    Path(chpt_path).mkdir(parents=True, exist_ok=True)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    if verbose is False:
        pbar = tqdm(total=len(trainloader))
    for epoch in range(n_epochs):
        generator.train()
        for i_batch, batch in enumerate(trainloader):
            frame0, frame1, pose1 = batch
            frame0, frame1, pose1 = frame0.to(
                device_ids[0]), frame1.to(device_ids[0]), pose1.to(device_ids[0])
            optim.zero_grad()
            #
            preds = generator(
                torch.cat([frame0, pose1], axis=1).to(device_ids[0]))
            # print(predictions.shape)
            # print(predictions, lbls)
            if faceid is not None:
                with torch.no_grad():
                    t_embeddings = faceid(frame1)
                p_embeddings = faceid(preds)
                loss = criterion(preds, frame1) + face_criterion(p_embeddings,
                                                                 t_embeddings, torch.ones((1), requires_grad=False).to(device_ids[0]))
            else:
                loss = criterion(preds, frame1)
            loss.backward()
            optim.step()
            meter.update(loss.item(), frame0.shape[0])
            pbar.update(1)
        if verbose is False:
            pbar.refresh()
            pbar.reset()
        val_loss = test_p2p(generator, testloader, criterion,
                            '{}/sammple_{}.jpg'.format(log_path, epoch), device_ids)
        if (epoch+1) % e_saves == 0:
            save_checkpoint(
                generator, '{}/checkpoint_{}.pth.tar'.format(chpt_path, epoch))
        avg_train.append(meter.avg)
        avg_val.append(val_loss)
        print('Training loss {} Val loss {}'.format(meter.avg, val_loss))
        plt.plot(avg_train, label='Train loss')
        plt.plot(avg_val, label='Val loss')
        plt.legend()
        plt.savefig('{}/losses.jpg'.format(log_path))
        plt.close()
        if scheduler is not None:
            scheduler.step(meter.avg)
        meter.reset()
    pbar.close()
