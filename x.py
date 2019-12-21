from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
from tqdm import tqdm


def parse_args():
    x = ArgumentParser()
    x.add_argument('--exp', type=str, default='baseline')
    x.add_argument('--device', type=str, default='cuda:0')
    x.add_argument('--dataset', type=str, default='cifar10@data/')
    x.add_argument('--dl_workers', type=int, default=8)
    x.add_argument('--epochs', type=int, default=100)
    x.add_argument('--batch_size', type=int, default=256)
    x.add_argument('--tqdm', type=int, default=1)
    x.add_argument('--noise_dim', type=int, default=128)
    x.add_argument('--class_embed_dim', type=int, default=128)
    x.add_argument('--head_dim', type=int, default=128)
    x.add_argument('--clf_dim', type=int, default=128)
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
    return train_loader, val_loader, 10


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


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        shape = (x.shape[0],) + self.shape
        return x.view(*shape)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)


class Upsample2d(nn.Module):
    def __init__(self, mul):
        super().__init__()
        self.mul = mul

    def forward(self, x):
        return F.upsample_bilinear(x, self.mul)


class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, class_embed_dim, head_dim):
        super().__init__()

        k = head_dim * 2

        self.noise_head = nn.Sequential(
            nn.Linear(noise_dim, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(k, k),
        )

        self.class_head = nn.Sequential(
            nn.Embedding(num_classes, class_embed_dim),
            Flatten(),
            nn.Linear(class_embed_dim, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(k, k),
        )

        self.trunk = nn.Sequential(
            nn.Linear(k * 2, k * 2),
            nn.BatchNorm1d(k * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(k * 2, k * 2),
            Reshape(k, 2, 2),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            Upsample2d(2),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            Upsample2d(2),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            Upsample2d(2),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            Upsample2d(2),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise, classes):
        noise = self.noise_head(noise)
        classes = self.class_head(classes)
        x = torch.cat([noise, classes], 1)
        return self.trunk(x)


class Classifier(nn.Sequential):
    def __init__(self, dim, num_classes):
        k = dim
        super().__init__(
            nn.Conv2d(3, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, 1, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, 2, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, 2, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, 2, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, 3, 2, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            Flatten(),
            nn.Dropout(),
            nn.Linear(k * 4, k * 4),
            nn.BatchNorm1d(k * 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(k * 4, num_classes),
        )


def train_on_batch(clf, clf_optimizer, x, y_true):
    clf.train()
    clf.zero_grad()
    y_pred = clf(x)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    clf_optimizer.step()
    return (y_pred.max(1)[1] == y_true).sum().item()


def validate_on_batch(clf, x, y_true):
    clf.eval()
    y_pred = clf(x)
    return (y_pred.max(1)[1] == y_true).sum().item()


def main(args):
    device = torch.device(args.device)
    dataset_type, dataset_dir = args.dataset.split('@')
    t_loader, v_loader, num_classes = load_dataset(
        dataset_type, dataset_dir, args.batch_size, args.dl_workers)
    gen = Generator(args.noise_dim, num_classes, args.class_embed_dim, args.head_dim)
    gen.to(device)
    gen_optimizer = Adam(gen.parameters())
    clf = Classifier(args.clf_dim, num_classes)
    clf.to(device)
    clf_optimizer = Adam(clf.parameters())
    for epoch in range(args.epochs):
        t_ok = 0
        t_all = 0
        v_ok = 0
        v_all = 0
        for is_train, x, y_true in each_batch(t_loader, v_loader, args.tqdm, device):
            if is_train:
                t_ok += train_on_batch(clf, clf_optimizer, x, y_true)
                t_all += x.shape[0]
            else:
                v_ok += validate_on_batch(clf, x, y_true)
                v_all += x.shape[0]
        t = t_ok * 100.0 / t_all
        v = v_ok + 100.0 / v_all
        print('%4d %6.2f %6.2f' % (epoch, t, v))


if __name__ == '__main__':
    main(parse_args())
