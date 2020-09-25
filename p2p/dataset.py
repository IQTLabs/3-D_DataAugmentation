import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


__all__ = ['P2PDataset']


class P2PDataset(Dataset):
    def __init__(self, df=None, transform=None, data_path=''):
        self.df = df
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        dir_path = '{}/{}/{}'.format(self.data_path,
                                     entry['name'], entry['snippet'])
        index = np.random.choice([x for x in range(5)], 2, replace=False)
        frames = [self.transform(Image.open(
            '{}/frame_{}.jpg'.format(dir_path, x))) for x in index]
        pose = self.to_tensor(Image.open(
            '{}/pose_{}.jpg'.format(dir_path, index[-1])))

        return frames[0], frames[1], pose

    def __len__(self):
        return len(self.df)
