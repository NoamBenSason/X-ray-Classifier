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

from models import DPConvNet, ViTForClassification

DATA_DIR = "data/chest_xray/"


def train(config, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Going over batches")):
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


def test(model, device, test_loader, split):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_n = 1
        for data, target in tqdm(test_loader, desc="Going over test batches"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Batch {batch_n}, Loss: {test_loss}, correct count: {correct}")
            batch_n += 1

    test_loss /= len(test_loader.dataset)

    print(
        f"\n{split} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f})\n")

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


def get_config(dp, model_name):
    sweep_config = {}
    sweep_config['method'] = 'bayes'
    sweep_config['metric'] = {'name': 'avg_test_accuracy', 'goal': 'maximize'}
    param_dict = {
        'BATCH_SIZE': {'values': [32, 64]},
        'TEST_BATCH_SIZE': {'value': 128},
        # 'NUM_EPOCHS': {'value': 1},
        'NUM_EPOCHS': {'values': [5, 10]},
        'NUM_RUNS': {'value': 1},
        # "LR": {'distribution': 'uniform', 'min': 0.0001, 'max': 0.05}, # todo
        "LR": {'value': 0.00001},
        # 'EPSILON':{'value':1.0},
        'EPSILON': {'distribution': 'uniform', 'min': 0.5, 'max': 20},
        # 'C':{'value': 1.0},
        'C': {'distribution': 'uniform', 'min': 0.1, 'max': 2},
        # 'DELTA':{'value': 1e-05},
        'DELTA': {'distribution': 'uniform', 'min': 0.0000001, 'max': 0.0001},
        'DEVICE': {'value': "cuda"},
        'SAVE_MODEL': {'value': False},
        'DISABLE_DP': {'value': not dp},
        'OPTIMIZER': {'value': "adam"},
        # 'OPTIMIZER':{'values': ["sgd", "adam"]}, # todo
        # 'BASIC_TRANSFORM': {'values': [True, False]}, # todo
        'BASIC_TRANSFORM': {'value': False},
        # 'BASIC_TRANSFORM': {'values': [True, False]}, # todo
        'MODEL_NAME': {'value': model_name},
    }

    if model_name == "ViTForClassification":
        param_dict.update({
            "patch_size": {'value': 4},
            "hidden_size": {'value': 128},
            "num_hidden_layers": {'value': 4},
            "num_attention_heads": {'value': 4},
            "intermediate_size": {'value': 4 * 48},
            "hidden_dropout_prob": {'value': 0.0},
            "attention_probs_dropout_prob": {'value': 0.0},
            "initializer_range": {'value': 0.02},
            "image_size": {'value': 100},
            "num_classes": {'value': 4},
            "num_channels": {'value': 3},
            "qkv_bias": {'value': True},
        })

    time_str = time.strftime("%Y%m%d-%H%M%S")
    s = "DP" if dp else "no_DP"
    sweep_config['name'] = f"{model_name}_{s}_{time_str}"
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
            ImageFolder(DATA_DIR + "test",
                        transform=get_transforms(config)),  # todo maybe transform without augs?
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

        train_loss_over_epochs = np.zeros((config["NUM_RUNS"], config["NUM_EPOCHS"]))

        for run_i in range(config["NUM_RUNS"]):  # number of experiments

            if config["MODEL_NAME"] == "DPConvNet":
                model = DPConvNet().to(config["DEVICE"])
            elif config["MODEL_NAME"] == "ViTForClassification":
                model = ViTForClassification(config).to(config["DEVICE"])

            if config["OPTIMIZER"] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config["LR"])
            elif config["OPTIMIZER"] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config["LR"], momentum=0)
            privacy_engine = None

            if not config["DISABLE_DP"]:
                privacy_engine = PrivacyEngine()

                model, optimizer, train_dt = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    max_grad_norm=config["C"],
                    target_delta=config["DELTA"],
                    target_epsilon=config["EPSILON"],
                    epochs=config["NUM_EPOCHS"],
                )

            for epoch in range(1, config["NUM_EPOCHS"] + 1):
                train_loss_mid_train = train(config, model, config["DEVICE"], train_loader, optimizer, privacy_engine,
                                             epoch)
                train_loss_over_epochs[run_i][epoch - 1] = train_loss_mid_train
                wandb.log({"train_loss": train_loss_mid_train}, step=epoch)

                if epoch % 3 == 0:
                    test_accuracy, test_loss = test(model, config["DEVICE"], test_loader, split="Test")
                    wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss}, step=epoch)

            train_accuracy, train_loss = test(model, config["DEVICE"], train_loader, split="Train")
            test_accuracy, test_loss = test(model, config["DEVICE"], test_loader, split="Test")

            run_results_train_accuracy.append(train_accuracy)
            run_results_train_loss.append(train_loss)

            run_results_test_accuracy.append(test_accuracy)
            run_results_test_loss.append(test_loss)

        if len(run_results_test_accuracy) > 1:
            print(
                "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                    len(run_results_test_accuracy), np.mean(run_results_test_accuracy) * 100,
                                                    np.std(run_results_test_accuracy) * 100
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
    # Training settings
    parser = argparse.ArgumentParser(
        description="DP with Chest X-ray",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="cnn",
        help='model name - cnn of vit')

    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )

    args = parser.parse_args()

    if args.model_name == "vit":
        official_name = "ViTForClassification"
    elif args.model_name == "cnn":
        official_name = "DPConvNet"

    sweep_id = wandb.sweep(get_config(dp=not args.disable_dp, model_name=official_name),
                           project="advanced_privacy_project",
                           entity="noambs")

    wandb.agent(sweep_id, main, count=1000, project="advanced_privacy_project", entity="noambs")
