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
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

    avg_train_loss = np.mean(losses)
    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {avg_train_loss:.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) privacy budget spent so far"
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


def main():
    # python dp_chest_xray.py - -device = cuda - n = 10 - -lr = 0.001 - -sigma = 1.3 - c = 1.5 - b = 16
    print("DP with Chest X-ray")

    # Training settings
    parser = argparse.ArgumentParser(
        description="DP with Chest X-ray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config_name', type=str, help='config name')

    args = parser.parse_args()
    config_name = args.config_name

    with open(f"configs/{config_name}.json") as f:
        config = json.load(f)

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=config["BATCH_SIZE"],
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=config["TEST_BATCH_SIZE"],
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=config["NUM_EPOCHS"],
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=config["NUM_RUNS"],
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config["LR"],
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=config["SIGMA"],
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=config["C"],
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=config["DELTA"],
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config["DEVICE"],
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=config["SAVE_MODEL"],
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=config["DISABLE_DP"],
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=config["SECURE_RNG"],
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default=config["OPTIMIZER"],
        help='optimizer name')

    parser.add_argument(
        '--basic_transform',
        type=str,
        default=config["BASIC_TRANSFORM"],
        help='If to do basic transform or fancy transform')

    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR + "train", transform=get_transforms(config)),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,

    )
    test_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR + "test", transform=get_transforms(config)),
        batch_size=args.test_batch_size,
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

    train_loss_over_epochs = np.zeros((args.n_runs,args.epochs))

    for run_i in range(args.n_runs): # number of experiments
        model = DPConvNet().to(device)

        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else: # args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm, # C
            )
        for epoch in range(1, args.epochs + 1):
            train_loss_mid_train = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            train_loss_over_epochs[run_i][epoch-1] = train_loss_mid_train

        train_accuracy, train_loss = test(model, device, train_loader)
        test_accuracy, test_loss = test(model, device, test_loader)

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
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/results", exist_ok=True)
    os.makedirs("out/saved_models", exist_ok=True)
    time_str = time.strftime("%Y%m%d-%H%M%S")

    repro_str = (
        f"chest_xray_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results_test_accuracy, f"out/results/run_results_{config_name}_{time_str}.pt")

    with open(f"out/results/final_results_{config_name}_{time_str}.json", "w") as f:
        json.dump({
            "config_name": config_name,
            "config": config,
            "avg_train_loss": np.mean(run_results_train_loss),
            "avg_train_accuracy": np.mean(run_results_train_accuracy),
            "std_train_accuracy": np.std(run_results_train_accuracy),

            "avg_test_loss": np.mean(run_results_test_loss),
            "avg_test_accuracy": np.mean(run_results_test_accuracy),
            "std_test_accuracy": np.std(run_results_test_accuracy),

        },f)

    train_loss_over_epochs = train_loss_over_epochs.tolist()
    file_path = f"out/results/train_loss_over_time_{config_name}_{time_str}.json"
    json.dump(train_loss_over_epochs, codecs.open(file_path, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)

    if args.save_model:
        torch.save(model.state_dict(), f"out/saved_models/chest_xray_cnn_{config_name}_{time_str}.pt")


if __name__ == "__main__":
    main()
