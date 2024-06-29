"""
Train VFL on ModelNet-10 dataset
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.distributions.binomial import Binomial

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve, auc

import argparse
import numpy as np
import time
import pandas as pd
import os
import pickle
import math

from models.resnet2 import *
from models.resnet_top import *
from models.mvcnn import *
from models.mvcnn_top_small2 import *
from models.mvcnn_bottom_small2 import *
from custom_dataset import MultiViewDataSet
from models.tabular_bottom import tabular_bottom
from models.tabular_top import tabular_top

parser = argparse.ArgumentParser(description='PBM-VFL')
parser.add_argument('--data', type=str, default='cifar',
                    help='dataset (default: cifar)')
parser.add_argument('--num_clients', type=int, default=4,
                    help='number of clients (default: 4)')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs (default: 250)')
parser.add_argument('--batch-size', type=int, default=100,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--print-freq', type=int, default=10,
                    help='print frequency (default: 10)')
parser.add_argument('--seed', type=int, default=42,
                    help='randomization seed (default: 42)')
parser.add_argument('--resume', type=str, default='false',
                    help='resume existing training (default: false)')
parser.add_argument('--quant_bin', type=int, default=0,
                    help='number of quantization buckets')
parser.add_argument('--theta', type=float, default=0,
                    help='theta value (default: 0)')

# Parse input arguments
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
path = '/gpfs/u/home/VFLA/VFLAnrnl/scratch'

# PBM functions
def quantize(x, theta, m):
    p = torch.add(0.5, torch.mul(theta, x))
    binom = Binomial(m, p)
    noise = binom.sample()
    y = x.clone()
    y.data = noise
    return y

def dequantize(q, theta, m, n):
    det = torch.sub(q, m * n / 2)
    sum = torch.div(det, theta * m)
    return sum

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Load dataset and create models
train_loader, test_loader = None, None
num_classes, coords_per = 0, 0
models = []
optimizers = []

if args.data == 'mn10':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform1 =transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    transform2 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ])

    dset_train = MultiViewDataSet(f'{path}/datasets/10class/classes', 'train', transform=transform1)
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
    indices = torch.randperm(len(dset_train))
    dset_train_sub = torch.utils.data.Subset(dset_train, indices[:int(len(dset_train)/4)])
    train_loader = DataLoader(dset_train_sub, batch_size=args.batch_size, shuffle=False, num_workers=1)
    dset_val = MultiViewDataSet(f'{path}/datasets/10class/classes', 'test', transform=transform2)
    test_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)
    num_classes = len(dset_train.classes)
    coords_per = int(12/args.num_clients)

    for i in range(args.num_clients+1):
        if i == args.num_clients:
            model = mvcnn_top(pretrained=False, num_classes=num_classes, num_clients=1)
        else:
            model = mvcnn_bottom(pretrained=False,num_classes=num_classes)
        model.to(device)
        cudnn.benchmark = True
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        models.append(model)
        optimizers.append(optimizer)

elif args.data == 'cifar':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    dset_train = torchvision.datasets.CIFAR10(root=f'{path}/datasets/', train=True, download=False, transform=transform1)
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=False, num_workers=1)
    dset_val = torchvision.datasets.CIFAR10(root=f'{path}/datasets/', train=False, download=False, transform=transform1)
    test_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=1)
    num_classes = len(dset_train.classes)
    coords_per = 16

    for i in range(args.num_clients+1):
        if i == args.num_clients:
            model = resnet_top(pretrained=False, num_classes=num_classes, num_clients=1)
        else:
            model = ResNet18()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        model.to(device)
        cudnn.benchmark = True
        models.append(model)
        optimizers.append(optimizer)

elif args.data == 'activity':
    train_data = pd.read_csv(f'{path}/datasets/activity/train.csv', sep=',')
    test_data = pd.read_csv(f'{path}/datasets/activity/test.csv', sep=',')
    labels = train_data.iloc[:,-1].unique()
    num_classes = len(labels)
    coords_per = int(560/args.num_clients)
    label_nums = np.arange(0, len(labels), 1, dtype=int)
    replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
    train_data.iloc[:,-1] = train_data.iloc[:,-1].replace(replace_dict)
    train_data = train_data.to_numpy().astype('float')
    test_data.iloc[:,-1] = test_data.iloc[:,-1].replace(replace_dict)
    test_data = test_data.to_numpy().astype('float')

    # Create train/test split
    train_raw = (train_data[:, :560], train_data[:, -1])
    test_raw = (test_data[:-10, :560], test_data[:-10, -1]) # leave out 10 samples for auditing
    dset_train = TensorDataset(torch.Tensor(train_raw[0]), torch.Tensor(train_raw[1]))
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dset_val = TensorDataset(torch.Tensor(test_raw[0]), torch.Tensor(test_raw[1]))
    test_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=1)

    for i in range(args.num_clients+1):
        if i == args.num_clients:
            model = tabular_top(num_classes=num_classes, embedding_size=1)
        else:
            model = tabular_bottom(input_dim=coords_per, output_dim=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        model.to(device)
        cudnn.benchmark = True
        models.append(model)
        optimizers.append(optimizer)

elif args.data == 'phishing':
    data = pd.read_csv(f'{path}/datasets/phishing/phishing.csv', sep=',')
    labels = data.iloc[:,-1].unique()
    label_nums = np.arange(0, len(labels), 1, dtype=int)
    replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
    data.iloc[:,-1] = data.iloc[:,-1].replace(replace_dict)
    data = data.to_numpy().astype('float')
    data[:,:-1] = (data[:,:-1] - data[:,:-1].min())/(data[:,:-1].max() - data[:,:-1].min())

    # Create train/test split
    split_idx = int(len(data)*0.8)
    train_raw = (data[:split_idx, :30], data[:split_idx, -1])
    test_raw = (data[split_idx:, :30], data[split_idx:, -1])
    dset_train = TensorDataset(torch.Tensor(train_raw[0]), torch.Tensor(train_raw[1]))
    train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dset_val = TensorDataset(torch.Tensor(test_raw[0]), torch.Tensor(test_raw[1]))
    test_loader = DataLoader(dset_val, batch_size=args.batch_size, shuffle=True, num_workers=1)
    num_classes = 2
    coords_per = int(30/args.num_clients)

    for i in range(args.num_clients+1):
        if i == args.num_clients:
            model = tabular_top(num_classes=num_classes)
        else:
            model = tabular_bottom(input_dim=coords_per)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        model.to(device)
        cudnn.benchmark = True
        models.append(model)
        optimizers.append(optimizer)
elif args.data == 'imagenet':
    pass

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
best_loss = 0.0
start_epoch = 0

losses = []
accs_train = []
accs_test = []

# optionally resume from a checkpoint
if args.resume == 'true':
    for i in range(args.num_clients+1):
        cpfile = os.path.join(f'{path}/checkpoints/{args.data}/{i}_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pth.tar')
        if os.path.isfile(cpfile):
            print("=> loading checkpoint")
            checkpoint = torch.load(cpfile, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            start_epoch = checkpoint["epoch"]
            models[i].load_state_dict(checkpoint["state_dict"])
            optimizers[i].load_state_dict(checkpoint["optimizer"])
            losses = pickle.load(open(f'{path}/results/{args.data}/loss_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            accs_train = pickle.load(open(f'{path}/results/{args.data}/accs_train_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            accs_test = pickle.load(open(f'{path}/results/{args.data}/accs_test_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'rb'))
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found")

def save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, step):
    """
    Evaluate and save current loss and accuracy
    """
    avg_train_acc, avg_loss = eval(models, train_loader)
    avg_test_acc, _ = eval(models, test_loader)

    losses.append(avg_loss)
    accs_train.append(avg_train_acc)
    accs_test.append(avg_test_acc)

    pickle.dump(losses, open(f'{path}/results/{args.data}/loss_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'wb'))
    pickle.dump(accs_train, open(f'{path}/results/{args.data}/accs_train_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'wb'))
    pickle.dump(accs_test, open(f'{path}/results/{args.data}/accs_test_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pkl', 'wb'))

    print('Iter [%d/%d]: Test Acc: %.2f - Train Acc: %.2f - Loss: %.4f' 
            % (step + 1, len(train_loader), avg_test_acc.item(), avg_train_acc.item(), avg_loss))

def train(models, optimizers):
    """
    Train all clients on all batches 
    """

    train_size = len(train_loader)
    server_model = models[-1]

    Hs = np.empty((len(train_loader), args.num_clients), dtype=object)
    Hs.fill([])
    grads_Hs = np.empty((args.num_clients), dtype=object)
    grads_Hs.fill([])

    for step, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)
        
        inputs = torch.from_numpy(inputs)

        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)
        # Exchange embeddings
        H_orig = [None] * args.num_clients
        with torch.no_grad():
            H_nograd = [None] * args.num_clients

        for i in range(args.num_clients):
            x_local = None
            if args.data == 'mn10':
                x_local = inputs[:,coords_per*i:coords_per*(i+1),:,:,:]
            elif args.data == 'cifar':
                r = math.floor(i/2)
                c = i % 2
                x_local = inputs[:,:,coords_per*r:coords_per*(r+1),coords_per*c:coords_per*(c+1)]
                x_local = torch.transpose(x_local,0,1)
            elif args.data in ['activity', 'phishing']:
                x_local = inputs[coords_per*i:coords_per*(i+1),:].transpose(0, 1)
            elif args.data == 'imagenet':
                pass

            H_orig[i] = models[i](x_local)
            with torch.no_grad():
                H_nograd[i] = models[i](x_local)

            # Compress embedding / Quantization
            if args.quant_bin > 0:
                H_nograd[i] = quantize(H_nograd[i], args.theta, args.quant_bin)

        # embedding summation
        with torch.no_grad():
            sum_nograd = torch.sum(torch.stack(H_nograd),axis=0)
        if args.quant_bin > 0:
            sum_nograd = dequantize(sum_nograd, args.theta, args.quant_bin, args.num_clients)
        
        sum_embedding = torch.sum(torch.stack(H_orig),axis=0)
        if args.quant_bin > 0:
            sum_embedding.data = sum_nograd
        
        outputs = server_model(sum_embedding)
        loss = criterion(outputs, targets.long())

        # compute gradient and do SGD step
        for i in range(args.num_clients+1):
            optimizers[i].zero_grad()
        loss.backward()
        for i in range(args.num_clients+1):
            optimizers[i].step()

        if (step + 1) % args.print_freq == 0:
            print("\tServer Iter [%d/%d] Loss: %.4f" % (step + 1, train_size, loss.item()))


# Validation and Testing
def eval(models, data_loader):
    """
    Calculate loss and accuracy for a given data_loader
    """
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    all_labels = []
    all_auprc = []

    for _, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # Get current embeddings
            H_new = [None] * args.num_clients
            for i in range(args.num_clients):
                x_local = None
                if args.data == 'mn10':
                    x_local = inputs[:,coords_per*i:coords_per*(i+1),:,:,:]
                elif args.data == 'cifar':
                    r = math.floor(i/2)
                    c = i % 2
                    x_local = inputs[:,:,coords_per*r:coords_per*(r+1),coords_per*c:coords_per*(c+1)]
                    x_local = torch.transpose(x_local,0,1)
                elif args.data in ['activity', 'phishing']:
                    x_local = inputs[coords_per*i:coords_per*(i+1),:].transpose(0, 1)
                elif args.data == 'imagenet':
                    pass
                H_new[i] = models[i](x_local)

            # compute output
            outputs = models[-1](torch.sum(torch.stack(H_new),axis=0))
            loss = criterion(outputs, targets.long())

            total_loss += loss.item()
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

            if args.data == 'phishing':
                all_labels.extend(targets.cpu().numpy())
                probabilities = torch.sigmoid(outputs)
                all_auprc.extend(probabilities[:, 1].cpu().numpy())
                precision, recall, _ = precision_recall_curve(all_labels, all_auprc)
                auprc = auc(recall, precision)

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    if args.data == 'phising':
        return auprc, avg_loss
    else:
        return avg_test_acc, avg_loss

# Get initial loss/accuracy
if start_epoch == 0:
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, start_epoch)
# Training / Eval loop
train_size = len(train_loader)
for epoch in range(start_epoch, args.epochs):
    print('\n-----------------------------------')
    print(f'Clients: {args.num_clients}, Quant_bin: {args.quant_bin}, Theta: {args.theta}')
    print('Epoch: [%d/%d]' % (epoch+1, args.epochs))
    start = time.time()

    train(models, optimizers)
    
    print('Time taken: %.2f sec.' % (time.time() - start))
    save_eval(models, train_loader, test_loader, losses, accs_train, accs_test, epoch)

    # Checkpoints
    for i in range(args.num_clients+1):
        save_filename = os.path.join(f'{path}/checkpoints/{args.data}/{i}_seed{args.seed}_clients{args.num_clients}_quant{args.quant_bin}_theta{args.theta}.pth.tar')
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": models[i].state_dict(),
                "optimizer": optimizers[i].state_dict(),
            },
            filename=save_filename,
        )
        print(f"saved to '{save_filename}'")
