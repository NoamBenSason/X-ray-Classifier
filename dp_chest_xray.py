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

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

DATA_DIR = "data/chest_xray/"

# class XrayDataset(Dataset):
#     def __init__(self, train = True,transform=None, target_transform=None):
#         self.train = train
#         self.img_dir = "/data/chest_xray/train" if train else "/data/chest_xray/test"
#         self.transform = transform
#         self.target_transform = target_transform
#
#         self.label0_count = 50
#         self.label1_count = 40
#         self.label2_count = 30
#         self.label3_count = 40
#
#         self.label0 = "COVID19"
#         self.label1 = "NORMAL"
#         self.label2 = "PNEUMONIA"
#         self.label3 = "TUBERCULOSIS"
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#
#         if idx < self.label1_count:
#             img_path = os.path.join(self.img_dir,  self.label0)
#             image = read_image(img_path)
#             label = 0
#         elif idx < self.label1_count + self.label2_count:
#             img_path = os.path.join(self.img_dir, self.label1)
#             image = read_image(img_path)
#             label = 1
#         elif idx < self.label1_count + self.label2_count + self.label3_count:
#             img_path = os.path.join(self.img_dir,  self.label2)
#             image = read_image(img_path)
#             label = 2
#         else:
#             img_path = os.path.join(self.img_dir,  self.label3)
#
#             files = os.listdir(img_path)
#
#             img_path = os.path.join(img_path, )
#             image = read_image(img_path)
#             label = 3
#
#
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
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

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


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
    return correct / len(test_loader.dataset)


def get_transforms():
    train_transform = transforms.Compose([
        # transforms.RandomRotation(10),  # rotate +/- 10 degrees
        # transforms.RandomHorizontalFlip(),  # reverse 50% of images
        # transforms.Resize(100),  # resize shortest side
        # transforms.CenterCrop(100),  # crop longest side
        ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    return train_transform

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="DP with Chest X-ray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"$$ Current working directory: {os.getcwd()}")

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR+"train", transform=get_transforms()) ,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR+"test", transform=get_transforms()),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    for _ in range(args.n_runs):
        model = SampleConvNet().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
        run_results.append(test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"chest_xray_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"chest_xray_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()