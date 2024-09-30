import argparse
import time
import os
import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from opacus import PrivacyEngine

from dp_chest_xray import get_transforms
from models import DPConvNet, ViTForClassification

DATA_DIR = "data/chest_xray/"


def train(config, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Going over batches of epoch {epoch}")):
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
        wandb.log({"train_loss": avg_train_loss,"epsilon_spent_so_far": epsilon, "delta": config["DELTA"]  })
    else:
        print(f"Train Epoch: {epoch} \t Loss: {avg_train_loss:.6f}")
        wandb.log({"train_loss": avg_train_loss})

    return avg_train_loss


def test(model, device, test_loader, split, privacy_engine,config):
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
    epsilon = privacy_engine.accountant.get_epsilon(delta=config["DELTA"])
    print(
        f"\n{split} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}), eps = {epsilon}\n")

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy, test_loss




def get_config(dp, model_name):
    sweep_config = {}
    sweep_config['method'] = 'grid' if dp else "bayes"
    sweep_config['metric'] = {'name': 'avg_test_accuracy', 'goal': 'maximize'}
    param_dict = {
        'BATCH_SIZE': {'value': 32 if model_name == "vit" else 64},
        'TEST_BATCH_SIZE': {'value': 128},
        'NUM_EPOCHS': {'value': 10},
        'NUM_RUNS': {'value': 1},
        "LR": {'value': 0.00001 if model_name == "vit" else 0.001},
        'DEVICE': {'value': "cuda"},
        'SAVE_MODEL': {'value': False},
        'DISABLE_DP': {'value': not dp},
        'OPTIMIZER': {'value': "adam"},
        'BASIC_TRANSFORM': {'value': True},
        'MODEL_NAME': {'value': model_name},
    }

    if model_name == "ViTForClassification":
        param_dict.update({
            "patch_size": {'value': 4},
            "hidden_size": {'value': 128},
            "num_hidden_layers": {'value': 4},
            "num_attention_heads": {'value': 6},
            "intermediate_size": {'value': 4 * 128},
            "hidden_dropout_prob": {'value': 0.0},
            "attention_probs_dropout_prob": {'value': 0.0},
            "initializer_range": {'value': 0.02},
            "image_size": {'value': 100},
            "num_classes": {'value': 4},
            "num_channels": {'value': 3},
            "qkv_bias": {'value': True},
        })

    if dp:
        param_dict.update({

            'EPSILON': {'values': [0.25,0.5,1,3,8,12,20]},
            'C': {'values': [0.25,0.5,1,2,3]},
            'DELTA': {'values': [0.001,0.0001,0.00001,0.000001]},
        })

    time_str = time.strftime("%Y%m%d-%H%M%S")
    s = "DP" if dp else "no_DP"
    sweep_config['name'] = f"{model_name}_{s}_{time_str}"
    sweep_config['parameters'] = param_dict

    return sweep_config


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader = torch.utils.data.DataLoader(
            ImageFolder(DATA_DIR + "train", transform=get_transforms(config, split="train")),
            batch_size=config["BATCH_SIZE"],
            num_workers=0,
            pin_memory=True,
            shuffle=True,

        )
        test_loader = torch.utils.data.DataLoader(
            ImageFolder(DATA_DIR + "test",
                        transform=get_transforms(config, split="test")),
            batch_size=config["TEST_BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,

        )

        run_results_train_accuracy = []
        run_results_test_accuracy = []
        run_results_train_loss = []
        run_results_test_loss = []

        train_loss_over_epochs = np.zeros((config["NUM_RUNS"], config["NUM_EPOCHS"]))

        for run_i in range(config["NUM_RUNS"]):  # number of experiments

            if config["MODEL_NAME"] == "DPConvNet":
                model = DPConvNet().to(config["DEVICE"])
            elif config["MODEL_NAME"] == "ViTForClassification":
                sub_config =  {
                    "patch_size": config["patch_size"],
                    "hidden_size": config["hidden_size"],
                    "num_hidden_layers": config["num_hidden_layers"],
                    "num_attention_heads": config["num_attention_heads"],
                    "intermediate_size": config["intermediate_size"],
                    "hidden_dropout_prob": config["hidden_dropout_prob"],
                    "attention_probs_dropout_prob": config["attention_probs_dropout_prob"],
                    "initializer_range": config["initializer_range"],
                    "image_size": config["image_size"],
                    "num_classes": config["num_classes"],
                    "num_channels": config["num_channels"],
                    "qkv_bias": config["qkv_bias"],
                }
                model = ViTForClassification(sub_config).to(config["DEVICE"])

            if config["OPTIMIZER"] == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config["LR"])
            elif config["OPTIMIZER"] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config["LR"], momentum=0)
            privacy_engine = None

            if not config["DISABLE_DP"]:
                privacy_engine = PrivacyEngine()

                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
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

            if config["SAVE_MODEL"]:
                os.makedirs("out", exist_ok=True)
                time_str = time.strftime("%Y%m%d-%H%M%S")
                os.makedirs(f"out/{time_str}", exist_ok=True)

                torch.save(model.state_dict(),
                           f"out/{time_str}/chest_xray_{config["MODEL_NAME"]}_{time_str}.pt")

            train_accuracy, train_loss = test(model, config["DEVICE"], train_loader, split="Train",privacy_engine=privacy_engine,config=config)
            test_accuracy, test_loss = test(model, config["DEVICE"], test_loader, split="Test",privacy_engine=privacy_engine,config=config)

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
        '--sweep_number',
        type=str,
        default=None,
        help='option to provide already exist wandb sweep number')

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

    if args.sweep_number is None:
        sweep_id = wandb.sweep(get_config(dp=not args.disable_dp, model_name=official_name),
                               project="advanced_privacy_project",
                               entity="noambs")
    else:
        sweep_id = args.sweep_number

    wandb.agent(sweep_id, main, count=1000, project="advanced_privacy_project", entity="noambs")
