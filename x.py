from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
from tqdm import tqdm


def parse_args():
    x = ArgumentParser()
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--dataset', type=str, default='cifar10@data/')
    x.add_argument('--dl_workers', type=int, default=8)
    x.add_argument('--epochs', type=int, default=100)
    x.add_argument('--batch_size', type=int, default=256)
    x.add_argument('--tqdm', type=int, default=1)
    return x.parse_args()


def load_cifar10(dir_name, batch_size, num_workers):
    train_transform = tf.Compose([
        tf.RandomCrop(32, padding=4),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
    ])
    train_dataset = CIFAR10(root=dir_name, train=True, download=True,
                            transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_transform = tf.Compose([
        tf.ToTensor(),
    ])
    val_dataset = CIFAR10(root=dir_name, train=False, download=True,
                          transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    return train_loader, val_loader


def load_dataset(dataset_type, dataset_dir, batch_size, num_workers):
    if dataset_type == 'cifar10':
        return load_cifar10(dataset_dir, batch_size, num_workers)
    else:
        assert False


def each_split_batch(loader):
    for x in loader:
        yield x


def each_batch(t_loader, v_loader, use_tqdm, device):
    splits = [1] * len(t_loader) + [0] * len(v_loader)
    np.random.shuffle(splits)
    t = each_split_batch(t_loader)
    v = each_split_batch(v_loader)
    if use_tqdm:
        splits = tqdm(splits, leave=False)
    for is_train in splits:
        if is_train:
            each = t
        else:
            each = v
        x, y_true = next(each)
        x = x.to(device)
        y_true = y_true.to(device)
        yield is_train, x, y_true


def main(args):
    device = torch.device(args.device)
    dataset_type, dataset_dir = args.dataset.split('@')
    t_loader, v_loader = load_dataset(dataset_type, dataset_dir, args.batch_size,
                                      args.dl_workers)
    for epoch in range(args.epochs):
        for is_train, x, y_true in each_batch(t_loader, v_loader, args.tqdm, device):
            pass


if __name__ == '__main__':
    main(parse_args())
