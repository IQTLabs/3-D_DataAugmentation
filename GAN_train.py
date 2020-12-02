import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import p2p
from p2p.config import args
from p2p.utils import createOptim

losses = ['MSE', 'VGGLoss']

if __name__ == '__main__':
    #
    root_dir = args['root_dir']
    log_dir = '{}/{}'.format(root_dir, args['log_dir'])
    p2p_df = pd.read_csv('{}/{}'.format(root_dir, args['meta_data']))
    p2p_df = p2p_df[p2p_df['n_frames'] == 10]
    train = p2p_df[p2p_df['set'] == 'train']
    test = p2p_df[p2p_df['set'] == 'test']
    #
    # imagenet_norm = {'mean': [0.485, 0.456, 0.406],
    #                 'std': [0.229, 0.224, 0.225]}
    frame_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args['norm_mean'],
                             std=args['norm_std'])
    ])
    data_path = '{}/{}'.format(root_dir, args['data_dir'])
    trainset = p2p.P2PDataset(
        df=train, transform=frame_transform, data_path=data_path)
    trainloader = DataLoader(trainset, batch_size=args['batch_size'],
                             shuffle=True, num_workers=args['n_workers'])
    testset = p2p.P2PDataset(
        df=test, transform=frame_transform, data_path=data_path)
    testloader = DataLoader(testset, batch_size=args['batch_size'],
                            shuffle=True, num_workers=args['batch_size'])
    #
    chpt_path = '{}/{}'.format(root_dir, args['log_dir'])
    netG = p2p.GlobalGenerator(input_nc=6, output_nc=3)
    netD = p2p.Discriminator()

    device_ids = list(args['gpu'])
    if len(args['gpu']) > 1:
        print('Data Parallel on {}'.format(args['gpu']))
        netG = nn.DataParallel(netG, device_ids=device_ids).to(device_ids[0])
        netD = nn.DataParallel(netD, device_ids=device_ids).to(device_ids[0])

    assert args['loss'] in losses, 'Missing loss implementation'

    if args['loss'] == 'MSE':
        loss = nn.MSELoss()
    elif args['loss'] == 'VGGLoss':
        loss = p2p.VGGLoss(device_ids[0], args['use_mse'])
        
    g_opt, g_sched = createOptim(parameters=netG.parameters(), lr=0.001)
    d_opt, d_sched = createOptim(parameters=netD.parameters(), lr=0.001)
    p2p.GAN_train_p2p(generator=netG, discriminator=netD, trainloader=trainloader,
                      testloader=testloader, g_opt=g_opt, g_sched=g_sched,
                      g_criterion=loss, d_opt=d_opt, d_sched=d_sched,
                      d_criterion=nn.BCEWithLogitsLoss(), GAN_weight=args['GAN_weight'],
                      n_epochs=args['n_epochs'], e_saves=args['e_saves'],
                      save_path=log_dir, device_ids=device_ids)
