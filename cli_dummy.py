
from dp_chest_xray import test, DATA_DIR
import torch
from models import DPConvNet
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import os

PATH ="out/config7_20240921-165329/chest_xray_cnn_config7_20240921-165329.pt"

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        ToTensor(),
    ])


    test_loader = torch.utils.data.DataLoader(
        ImageFolder(DATA_DIR + "test", transform=transform),
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,

    )

    model = DPConvNet().to("cuda")
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

    test_accuracy, test_loss = test(model, "cuda", test_loader, split="Test")

    print(f"test_loss: {test_loss}, test_accuracy: {test_accuracy}")

    # parser = argparse.ArgumentParser(
    #     description="DP with Chest X-ray",
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )
    #
    # parser.add_argument(
    #     '--model_name',
    #     type=str,
    #     default="cnn",
    #     help='model name - cnn of vit')
    #
    # parser.add_argument(
    #     "--disable-dp",
    #     action="store_true",
    #     default=True,
    #     help="Disable privacy training and just train with vanilla SGD",
    # )
    #
    # args = parser.parse_args()
    #
    # a = args.model_name
    # b = args.disable_dp
    # c =1

