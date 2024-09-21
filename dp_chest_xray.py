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
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from models import DPConvNet, ViTForClassification

DATA_DIR = "data/chest_xray/"


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader, desc = "Going over batches")):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch}, batch {_batch_idx}, Loss: {loss.item()}")

    avg_train_loss = np.mean(losses)
    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {avg_train_loss:.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta}) privacy budget spent so far"
        )
    else:
        print(f"Train Epoch: {epoch} \t Avg epoch loss: {avg_train_loss:.6f}")

    return avg_train_loss


def test(model, device, test_loader, split):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_n=1
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            print(f"Batch {batch_n}, Loss: {test_loss}, correct count: {correct}")
            batch_n += 1

    test_loss /= len(test_loader.dataset)

    print(
        f"\n{split} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f})\n")

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy, test_loss


def get_transforms(config, split = "train"):

    if split == "train":
        if not config["BASIC_TRANSFORM"]:
            transform = transforms.Compose([
                transforms.RandomRotation(10),  # rotate +/- 10 degrees
                transforms.RandomHorizontalFlip(),  # reverse 50% of images
                transforms.Resize(100),  # resize shortest side
                transforms.CenterCrop(100),  # crop longest side
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                ToTensor(),
            ])
    else:
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            ToTensor(),
        ])

    return transform


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
        "--epsilon",
        type=float,
        default=config["EPSILON"],
        help="epsilon for budget",
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
    # parser.add_argument(
    #     "--secure-rng",
    #     action="store_true",
    #     default=config["SECURE_RNG"],
    #     help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    # )

    parser.add_argument(
        "--basic_transform",
        action="store_true",
        default=config["BASIC_TRANSFORM"],
        help="If to do basic transform or fancy transform",
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default=config["OPTIMIZER"],
        help='optimizer name')

    parser.add_argument(
        '--model_name',
        type=str,
        default=config["MODEL_NAME"],
        help='model name - cnn of vit')

    args = parser.parse_args()
    if args.model_name == "vit":
        official_model_name = "ViTForClassification"
    elif args.model_name == "cnn":
        official_model_name = "DPConvNet"



    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR + "train", transform=get_transforms(config)),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,

    )
    test_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR + "test", transform=get_transforms(config, split="test")),
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

    os.makedirs("out", exist_ok=True)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"out/{config_name}_{time_str}", exist_ok=True)

    train_loss_over_epochs = np.zeros((args.n_runs, args.epochs))

    for run_i in range(args.n_runs):  # number of experiments

        if official_model_name  == "DPConvNet":
            model = DPConvNet().to(args.device)
        elif official_model_name  == "ViTForClassification":
            model = ViTForClassification(config).to(args.device)

        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine()

            model, optimizer, train_dt = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=args.max_grad_norm,
                target_delta=args.delta,
                target_epsilon=args.epsilon,
                epochs=args.epochs,
            )
        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch}")
            train_loss_mid_train = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            train_loss_over_epochs[run_i][epoch - 1] = train_loss_mid_train

            if epoch % 3 == 0:
                test_accuracy, test_loss = test(model, args.device, test_loader, split="Test")
                print(f"test_accuracy{test_accuracy}, test_loss {test_loss}")


        if args.save_model:
            torch.save(model.state_dict(), f"out/{config_name}_{time_str}/chest_xray_cnn_{config_name}_{time_str}.pt")

        train_accuracy, train_loss = test(model, device, train_loader, split="Train")
        test_accuracy, test_loss = test(model, device, test_loader, split="Test")

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


    with open(f"out/{config_name}_{time_str}/final_results_{config_name}_{time_str}.json", "w") as f:
        json.dump({
            "config_name": config_name,
            "config": config,
            "avg_train_loss": np.mean(run_results_train_loss),
            "avg_train_accuracy": np.mean(run_results_train_accuracy),
            "std_train_accuracy": np.std(run_results_train_accuracy),

            "avg_test_loss": np.mean(run_results_test_loss),
            "avg_test_accuracy": np.mean(run_results_test_accuracy),
            "std_test_accuracy": np.std(run_results_test_accuracy),
            "run_results_test_accuracy": run_results_test_accuracy
        }, f, indent=2)




if __name__ == "__main__":
    main()
