from __future__ import print_function
import argparse
from collections import OrderedDict

import torch
import spconv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

forward_time = 0
bw_time = 0
train_time = 0
test_time = 0

sparsity_cov1 = (0, 0)
sparsity_cov2 = (0, 0)
sparsity_relu1 = (0, 0)
sparsity_relu2 = (0, 0)
sparsity_maxpool = (0, 0)


class Net(nn.Module):
    def __init__(self, layer_names):

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.layer_map = {
            "td" : spconv.ToDense(),
            "sconv1" : spconv.SparseConv2d(1, 32, 3, 1),
            "sconv2" : spconv.SparseConv2d(32, 64, 3, 1),
            "dconv1" : nn.Conv2d(1, 32, 3, 1),
            "dconv2" : nn.Conv2d(32, 64, 3, 1),
            "smaxpool" : spconv.SparseMaxPool2d(2, 2),
            "dmaxpool" : nn.MaxPool2d((2,2)),
            "sbatch" : nn.BatchNorm1d(1),
            "dbatch" : nn.BatchNorm2d(1),
            "srelu" : nn.ReLU(),
            "drelu" : nn.ReLU(),
            "ts"    : None
        }

        self.layer_names = layer_names
        self.layers = self.generate_layers(layer_names)


    def generate_layers(self, names):
        layers = []
        for n in names:
            layers += self.layer_map[n]
        return layers


    def get_new_spar(self, news, old_spar_cnt):
        (olds, oldcnt) = old_spar_cnt
        cnt = oldcnt + 1
        news = (olds * oldcnt + news) / cnt
        return (news, cnt)

    def forward(self, x: torch.Tensor):
        global sparsity_cov1
        global sparsity_cov2
        global sparsity_relu1
        global sparsity_relu2
        global sparsity_maxpool

        for i, fn in enumerate(self.layer_names):
            if fn in ["srelu", "sbatch"]:
                x.features = self.layers[i](x.features)
            if fn == "ts":
                x = spconv.from_dense()
            else:
                x = self.layers[i](x)

        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    global forward_time
    global bw_time
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        forward_start = time.time()
        output = model(data)
        forward_end = time.time()
        loss = F.nll_loss(output, target)
        bw_start = time.time()
        loss.backward()
        bw_end = time.time()

        forward_time += forward_end - forward_start
        bw_time += bw_end - bw_start

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    global train_time
    global test_time
    global forward_time
    global bw_time
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # here we remove norm to get sparse tensor with lots of zeros
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           # here we remove norm to get sparse tensor with lots of zeros
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    layer_names = ["dbatch", "dconv1", "drelu", "dconv2", "drelu", "dmaxpool"]

    model = Net(layer_names).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        train_end = time.time()
        test_start = time.time()
        test(model, device, test_loader)
        test_end = time.time()
        train_time += train_end - train_start
        test_time += test_end - test_start
        scheduler.step()

        print("Sparsity cov1", sparsity_cov1)
        print("Sparsity relu1", sparsity_relu1)
        print("Sparsity cov2", sparsity_cov2)
        print("Sparsity relu2", sparsity_relu2)
        print("Sparsity max pool", sparsity_maxpool)
        print("Train time:", train_time, " Test time:", test_time)
        print("Forward time:", forward_time, " Backward time:", bw_time)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()