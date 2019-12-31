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
        # print(self.input_shape.channels)
        self.normaliseConv1 = nn.BatchNorm2d(
            num_features=32,
            affine = True,
        )

        self.normaliseConv2 = nn.BatchNorm2d(
            num_features=32,
            affine = True,
        )

        self.normaliseConv3 = nn.BatchNorm2d(
            num_features=64,
            affine = True,
        )

        self.normaliseConv4 = nn.BatchNorm2d(
            num_features=64,
            affine = True,
        )


        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            padding=(1,1),
            kernel_size=(3, 3),


        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            padding=(1,1),
            kernel_size=(3, 3),


        )
        self.initialise_layer(self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            padding=(1,1),
            kernel_size=(3, 3),


        )
        self.initialise_layer(self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            padding=(1,1),
            kernel_size=(3, 3),

        )
        self.initialise_layer(self.conv4)

        self.fc1 = nn.Linear(53760, 1024, bias = True)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(1024, 10, bias = True)
        self.initialise_layer(self.fc2)







    def forward(self, sounds: torch.Tensor) -> torch.Tensor:
        # print(sounds.size())
        #print(self.normaliseConv1(self.conv1(sounds)).size())
        #print(self.bias1.size())
        x = F.relu(self.normaliseConv1(self.conv1(sounds)))


        x = F.relu(self.normaliseConv2(self.conv2(self.dropout(x))))
        x = self.pool1(x)


        x = F.relu(self.normaliseConv3(self.conv3(x)))
        x = F.relu(self.normaliseConv4(self.conv4(self.dropout(x))))
        x = torch.flatten(x, 1)



        x = torch.sigmoid((self.fc1(self.dropout(x))))



        x = self.fc2(x)



        return x

    @staticmethod
    def initialise_layer(layer):
        #if hasattr(layer, "bias"):
        #    nn.init.zeros_(layer.bias)
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
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.step = 0
    # TODO
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for batch, labels, fname in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                data_load_end_time = time.time()


                ## TASK 1: Compute the forward pass of the model, print the output shape
                ##         and quit the program
                logits = self.model.forward(batch)

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

                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            # self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )


    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, fname in self.val_loader:
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

    nameArray = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

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


    # for i in range(train_loader.__len__()):
    #     print(train_loader.dataset.__getitem__(i)[0].size())


    model = CNN(height=85, width=41, channels=1, class_count=10, dropout=0.5)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-3)

    # print(train_dataset.__getitem__(0)[0].size())

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, DEVICE
    )

    trainer.train(
        epochs=50,
        print_frequency=10,
        val_frequency=2,
    )

    for i, (input, target, filename) in enumerate(train_loader):
        pass
    #           training code



    for i, (input, target, filename) in enumerate(val_loader):
        pass
    #           validation code


main()
