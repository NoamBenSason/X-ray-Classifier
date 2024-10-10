# Advanced Topics in Online Privacy and Cyber-security (67515) - Final Project
### Noam Ben Sason - 318505260

This is an instruction file on how to train and evaluate the project's models on a university computer. 
This includes the CNN model and the ViT model, both with or without differential privacy.
Note that for training with the wandb tool, one would need a wandb account, so I will not be providing 
instructions on how to run the models with wandb (this was done with a different file attached).

## Preparing the project:

1. Download the project directory to the computer.
2. Please make sure your python version is 3.9.2.
2. Create a virtual environment: `virtualenv dpvenv`
2. Activate the virtual environment:  `source dpvenv/bin/activate.csh`
3. Install all packages required: `pip install -r requirements.txt`
4. If there is cuda available, load it: `module load cuda`

## Download the data:

You can download the chest X-ray data from Kaggle:
https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis/data

If you want to download the data using a command line, you can follow the instructions here, using the dataset's source and name `jtiptj/chest-xray-pneumoniacovid19tuberculosis`:
https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/

After all the files are downloaded, please make sure all the data is in a folder named 'data', and all the splits of the data is in a folder inside it named 'chest_xray` (example below).
This folder should be in the project's directory, and the hierarchy need to be like this:

advanced_privacy_project/data/chest_xray/train

advanced_privacy_project/data/chest_xray/val

advanced_privacy_project/data/chest_xray/test

## Recurses: 

It is important to note that running a deep neural network is a lot more faster with a GPU instead of a CPU.
If there is a GPU available, It will be used.

If you have access to the phoenix cluster, you can ask for an interactive node with an available GPU:

`srun --mem=32gb -c4 --gres=gpu:1,vmem:24g --time=2:00:00 --pty $SHELL`

you will get a srun interactive note to run the code on.

## Running the model

To train and evaluate the model there are 2 options:

1. `python dp_chest_xray.py --config_name "<config_name>"` 

where `config_name` can be either "cnn_config" or "vit_config". Those existing configs includes the default
training hyperparameters for each model.

2. `python dp_chest_xray.py --model_name <model_name> --batch-size <batch_size> --test-batch-size <test_batch_size> --epochs <epoches_num> --lr <learning_rate> ----save-model --disable-dp --optimizer <optimizer_type> --epsilon <epsilon> -c <C> --delta <delta>`

   1. --model_name: The name of the model to train & evaluate. Can be "cnn" or "vit".
   2. --batch-size: The batch size to use in training time
   3. --test-batch-size: The batch size to use in evaluation time
   4. --epochs: Number of epochs to train on
   5. --lr: The learning rate od the training of the model.
   6. --save-model:  A flag indicating if you want to save the trained model.
   7. --disable-dp: A flag indicating if you want to disable the differential privacy mechanism. If this flag is in use, the DP mechanism will not be used.
   8. --optimizer: The optimizer type you want to use while training. Can be "sgd" or "adam".
   9. --epsilon: Epsilon for budget
   10. -c: Clip per-sample gradients to this norm
   11. --delta: Target delta

After the training and evaluation of the model, a directory name 'out' will be created, and the results will be saves to a json file in a directory in the 'out' directory.