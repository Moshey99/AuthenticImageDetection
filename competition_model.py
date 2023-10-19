"""Define your architecture here."""
import torch
from torch import nn,optim
import argparse
from models import SimpleNet,get_xception_based_model

from utils import load_dataset, load_model
import torch.nn.functional as F
from trainer import LoggingParameters, Trainer

# class myNet(nn.Module):
#     """Simple Convolutional and Fully Connect network."""
#
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
#         self.conv3 = nn.Conv2d(16, 24, kernel_size=5, stride=1, padding=2)
#         self.fc1 = nn.Linear(24 * 26 * 26, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 2)
#
#     def forward(self, image):
#         """Compute a forward pass."""
#         first_conv_features = self.pool(F.relu(self.conv1(image)))
#         second_conv_features = self.pool(F.relu(self.conv2(
#             first_conv_features)))
#         third_conv_features = self.pool(F.relu(self.conv3(
#             second_conv_features)))
#         # flatten all dimensions except batch
#         flattened_features = torch.flatten(third_conv_features, 1)
#         fully_connected_first_out = F.relu(self.fc1(flattened_features))
#         fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
#         two_way_output = self.fc3(fully_connected_second_out)
#         return two_way_output
#
# Arguments
def parse_args():
    """Parse script arguments.

    Get training hyper-parameters such as: learning rate, momentum,
    batch size, number of training epochs and optimizer.
    Get training dataset and the model name.
    """
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum')
    parser.add_argument('--batch_size', '-b', default=32, type=int,
                        help='Training batch size')
    parser.add_argument('--epochs', '-e', default=4, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--optimizer', '-o', default='Adam', type=str,
                        help='Optimization Algorithm')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()

def train_my_competition_model():
    args = parse_args()
    # Data
    print(f'==> Preparing data: {args.dataset.replace("_", " ")}..')

    train_dataset = load_dataset(dataset_name=args.dataset,
                                 dataset_part='train')
    val_dataset = load_dataset(dataset_name=args.dataset, dataset_part='val')
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    # Model
    model = get_xception_based_model()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Build optimizer
    optimizers = {
        'SGD': lambda: optim.SGD(model.parameters(),
                                 lr=args.lr,
                                 momentum=args.momentum),
        'Adam': lambda: optim.Adam(model.parameters(), lr=args.lr),
    }

    optimizer_name = args.optimizer
    if optimizer_name not in optimizers:
        raise ValueError(f'Invalid Optimizer name: {optimizer_name}')
    print(f"Building optimizer {optimizer_name}...")
    optimizer = optimizers[args.optimizer]()
    print(optimizer)

    optimizer_params = optimizer.param_groups[0].copy()
    # remove the parameter values from the optimizer parameters for a cleaner
    # log
    del optimizer_params['params']

    # Batch size
    batch_size = args.batch_size

    # Training Logging Parameters
    logging_parameters = LoggingParameters(model_name='myNet',
                                           dataset_name=args.dataset,
                                           optimizer_name=optimizer_name,
                                           optimizer_params=optimizer_params,)

    # Create an abstract trainer to train the model with the data and parameters
    # above:
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      batch_size=batch_size,
                      train_dataset=train_dataset,
                      validation_dataset=val_dataset,
                      test_dataset=test_dataset)
    # Train, evaluate and test the model:
    trainer.run(epochs=args.epochs, logging_parameters=logging_parameters)


def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = get_xception_based_model()


    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition.pt')['model'])
    return model

if __name__=="__main__":
    train_my_competition_model()

