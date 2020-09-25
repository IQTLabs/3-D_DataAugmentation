import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import p2p
from p2p.config import args
from p2p.utils import createOptim

if __name__ == '__main__':
    #
    root_dir = args['root_dir']
    log_dir = '{}/{}'.format(root_dir, args['log_dir'])
    p2p_df = pd.read_csv('{}/{}'.format(root_dir, args['meta_data']))
    p2p_df = p2p_df[p2p_df['n_frames'] == 10]
    train = p2p_df[p2p_df['set'] == 'train']
    test = p2p_df[p2p_df['set'] == 'test']
    #
    imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}
    frame_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_norm['mean'],
                             std=imagenet_norm['std'])
    ])
    data_path = '{}/{}'.format(root_dir, args['data_dir'])
    trainset = p2p.P2PDataset(
        df=train, transform=frame_transform, data_path=data_path)
    trainloader = DataLoader(trainset, batch_size=args['batch_size'],
                             shuffle=True, num_workers=args['batch_size'])
    testset = p2p.P2PDataset(
        df=test, transform=frame_transform, data_path=data_path)
    testloader = DataLoader(testset, batch_size=args['batch_size'],
                            shuffle=True, num_workers=args['batch_size'])
    #
    chpt_path = '{}/{}'.format(root_dir, args['log_dir'])
    netG = p2p.GlobalGenerator(input_nc=6, output_nc=3)
    if args['face_id']:
        faceid = p2p.VoxFaceID(
            pretrain_path='{}/p2p/pretrained/pretrained_VoxFaceID.pth.tar'.format(root_dir))
    else:
        faceid = None

    device_ids = list(args['gpu'])
    if len(args['gpu']) > 1:
        print('Data Parallel on {}'.format(args['gpu']))
        netG = nn.DataParallel(netG, device_ids=device_ids).to(device_ids[0])
        faceid = nn.DataParallel(
            faceid, device_ids=device_ids).to(device_ids[0])

    parameters = list(netG.parameters())
    optimizer, scheduler = createOptim(parameters=parameters, lr=0.001)
    p2p.train_p2p(generator=netG, faceid=faceid, trainloader=trainloader,
                  testloader=testloader, optim=optimizer, scheduler=scheduler,
                  criterion=nn.MSELoss(), n_epochs=args['n_epochs'],
                  e_saves=args['e_saves'], save_path=log_dir, device_ids=device_ids)
