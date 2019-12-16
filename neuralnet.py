from dataset import UrbanSound8KDataset
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout(p=dropout)

        self.normaliseConv1 = nn.BatchNorm2d(
            num_features=32
        )
        self.normaliseConv2 = nn.BatchNorm2d(
            num_features=32
        )
        self.normaliseConv3 = nn.BatchNorm2d(
            num_features=64
        )
        self.normaliseConv4 = nn.BatchNorm2d(
            num_features=64
        )

        self.conv1 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2,2)
        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2,2)
        )
        self.initialise_layer(self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2,2)
        )
        self.initialise_layer(self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(2,2)
        )
        self.initialise_layer(self.conv4)
        
        self.fc1 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc1)

    def forward(self, sounds: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.normaliseConv1(self.conv1(sounds)))
        
        x = F.relu(self.normaliseConv2(self.conv2(x)))
        x = self.pool2(x)
        # TASK 4: Flatten the output of the pooling layer so it is of shape
        ##         (batch_size, 4096)
        x = torch.flatten(x, 1)
        ## TASK 5-2: Pass x through the first fully connected layer
        x = F.relu(self.normaliseFC(self.fc1(self.dropout(x))))
        ## TASK 6-2: Pass x through the last fully connected layer
        x = self.fc2(self.dropout(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
    # TODO
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                logits = self.model.forward(batch)
                #print(output.shape)
                #import sys; sys.exit(1)


                ## TASK 7: Rename `output` to `logits`, remove the output shape printing
                ##         and get rid of the `import sys; sys.exit(1)`

                ## TASK 9: Compute the loss using self.criterion and
                ##         store it in a variable called `loss`
                #loss = torch.tensor(0)
                loss = self.criterion(logits, labels)

                ## TASK 10: Compute the backward pass

                loss.backward()


                ## TASK 12: Step the optimizer and then zero out the gradient buffers.

                self.optimizer.step()

                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)


        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """


    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)

def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    classLabel = [0] * 10
    classPred = [0] * 10
    
    # TODO
    nameArray = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    for i in range(len(labels)):
        if(labels[i] == preds[i]):
            classLabel[labels[i]] += 1
        classPred[preds[i]] += 1

    for i in range(10):
        acc = 0
        if(classPred[i]):
            acc = classLabel[i]/classPred[i]
        print(f"Class accuracy for {nameArray[i]}: {acc * 100:2.2f}")


def main():

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


    mode = "LMC"
    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', mode),
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', mode),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)


    for i in range(train_loader.__len__()):
        print(train_loader.dataset.__getitem__(i)[0].size())
    
    # model = CNN(height=32, width=32, channels=1, class_count=10, dropout=0.5)

    # criterion = nn.CrossEntropyLoss()


    # optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9)
    
    # trainer = Trainer(
    #     model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    # )

    # trainer.train(
    #     args.epochs,
    #     args.val_frequency,
    #     print_frequency=args.print_frequency,
    #     log_frequency=args.log_frequency,
    # )

    for i, (input, target, filename) in enumerate(train_loader):
        pass
    #           training code



    for i, (input, target, filename) in enumerate(val_loader):
        pass
    #           validation code


main()
