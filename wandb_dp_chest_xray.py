import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time
import json
import codecs
import os
import wandb
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DATA_DIR = "data/chest_xray/"


class DPConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(150, 200, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(200, 200, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(200, 250, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(250, 250, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(36000, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x of shape [B, 3, 300, 300]
        x = F.relu(self.conv1(x))  # -> [B, 100, 300, 300]
        x = F.relu(self.conv2(x))  # -> [B, 150, 100, 100]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 150, 150, 150]

        x = F.relu(self.conv3(x))  # -> [B, 200, 150, 150]
        x = F.relu(self.conv4(x))  # -> [B, 200, 150, 150]
        x = F.max_pool2d(x, 2, 2)# -> [B, 200, 75, 75]

        x = F.relu(self.conv5(x))# -> [B, 250, 75, 75]
        x = F.relu(self.conv6(x))# -> [B, 250, 75, 75]
        x = F.max_pool2d(x, 2, 2)# -> [B, 250, 37, 37]

        x = torch.flatten(x, start_dim=1)  # -> [B, 342250]
        x = F.relu(self.fc1(x))  # -> [B, 64]
        x = F.relu(self.fc2(x))  # -> [B, 32]
        x = F.relu(self.fc3(x))  # -> [B, 16]
        x = F.relu(self.fc4(x))  # -> [B, 8]
        x = self.dropout(x)
        x = self.fc5(x)  # -> [B, 4]
        return x

    def name(self):
        return "DPConvNet"


def train(config, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_train_loss = np.mean(losses)
    if not config["DISABLE_DP"]:
        epsilon = privacy_engine.accountant.get_epsilon(delta=config["DELTA"])
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {avg_train_loss:.6f} "
            f"(ε = {epsilon:.2f}, δ = {config["DELTA"]}) privacy budget spent so far"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {avg_train_loss:.6f}")

    return avg_train_loss

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy, test_loss


def get_transforms(config):
    if not config["BASIC_TRANSFORM"]:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(100),  # resize shortest side
            transforms.CenterCrop(100),  # crop longest side
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((100, 100)),
            ToTensor(),
        ])

    return train_transform

def get_config():
    sweep_config = {}
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = {'name': 'avg_test_accuracy', 'goal': 'maximize'}
    param_dict = {
        'BATCH_SIZE': {'values': [32,64]},
        'TEST_BATCH_SIZE': {'value': [128]},
        'NUM_EPOCHS': {'values': [20,25]},
        'NUM_RUNS': {'value': [1]},
        "LR": {'distribution': 'uniform', 'min': 0.0001, 'max': 0.05},
        'SIGMA':{'distribution': 'uniform', 'min': 0.0001, 'max': 0.05}, #todo
        'C':{'distribution': 'uniform', 'min': 0.0001, 'max': 0.05}, #todo
        'DELTA':{'distribution': 'uniform', 'min': 0.0001, 'max': 0.05}, #todo
        'DEVICE':{'value': ["cuda"]},
        'SAVE_MODEL':{'value': [False]},
        'SECURE_RNG':{'value': [False]}, #todo
        'DISABLE_DP':{'values': [True, False]},
        'OPTIMIZER':{'values': ["sgd", "adam"]},
        'BASIC_TRANSFORM': {'values': [True, False]},
    }
    time_str = time.strftime("%Y%m%d-%H%M%S")
    sweep_config['name'] = f"DP_CNN_{time_str}"
    sweep_config['parameters'] = param_dict

    return sweep_config

def main(config=None):
    with wandb.init(config=config):
        config = wandb.config

        train_loader = torch.utils.data.DataLoader(
            ImageFolder(DATA_DIR + "train", transform=get_transforms(config)),
            batch_size=config["BATCH_SIZE"],
            num_workers=0,
            pin_memory=True,

        )
        test_loader = torch.utils.data.DataLoader(
            ImageFolder(DATA_DIR + "test", transform=get_transforms(config)),
            batch_size=config["TEST_BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,

        )

        # example, _ = ImageFolder(DATA_DIR+"train", transform=get_transforms(config))[20]
        # plt.imshow(example.permute(1, 2, 0))
        # plt.show()

        run_results_train_accuracy = []
        run_results_test_accuracy = []
        run_results_train_loss = []
        run_results_test_loss = []

        train_loss_over_epochs = np.zeros((config["NUM_RUNS"],config["NUM_EPOCHS"]))

        for run_i in range(config["NUM_RUNS"]): # number of experiments
            model = DPConvNet().to(config["DEVICE"])

            if  config["OPTIMIZER"]== 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config["LR"])
            elif config["OPTIMIZER"] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config["LR"], momentum=0)
            privacy_engine = None

            if not config["DISABLE_DP"]:
                privacy_engine = PrivacyEngine(secure_mode=config["SECURE_RNG"])
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=config["SIGMA"],
                    max_grad_norm=config["C"], # C
                )
            for epoch in range(1,config["NUM_EPOCHS"] + 1):
                train_loss_mid_train = train(config, model, config["DEVICE"], train_loader, optimizer, privacy_engine, epoch)
                train_loss_over_epochs[run_i][epoch-1] = train_loss_mid_train

            train_accuracy, train_loss = test(model, config["DEVICE"], train_loader)
            test_accuracy, test_loss = test(model, config["DEVICE"], test_loader)

            run_results_train_accuracy.append(train_accuracy)
            run_results_train_loss.append(train_loss)

            run_results_test_accuracy.append(test_accuracy)
            run_results_test_loss.append(test_loss)

        if len(run_results_test_accuracy) > 1:
            print(
                "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                    len(run_results_test_accuracy), np.mean(run_results_test_accuracy) * 100, np.std(run_results_test_accuracy) * 100
                )
            )

        wandb.log({
            "avg_train_loss": np.mean(run_results_train_loss),
            "avg_train_accuracy": np.mean(run_results_train_accuracy),
            "std_train_accuracy": np.std(run_results_train_accuracy),

            "avg_test_loss": np.mean(run_results_test_loss),
            "avg_test_accuracy": np.mean(run_results_test_accuracy),
            "std_test_accuracy": np.std(run_results_test_accuracy),
        })



if __name__ == "__main__":
    sweep_id = wandb.sweep(get_config(), project="advanced_privacy_project",
                           entity="noam-bs97")
    wandb.agent(sweep_id, main, count=1000)
