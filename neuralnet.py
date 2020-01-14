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
import copy
import sys
import argparse
from pathlib import Path

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float, isMLMC):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        self.dropout = nn.Dropout2d(p=dropout)
        self.normaliseConv1 = nn.BatchNorm2d(
            num_features=32,
        )

        self.normaliseConv2 = nn.BatchNorm2d(
            num_features=32,
        # if hasattr(layer, "bias"):
        #     nn.init.zeros_(layer.bias)
        )

        self.normaliseConv3 = nn.BatchNorm2d(
            num_features=64,
        )

        self.normaliseConv4 = nn.BatchNorm2d(
            num_features=64,
        )


        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            padding=(1,1),
            kernel_size=(3, 3),
            bias=False,
            stride=(1,1)


        )
        self.initialise_layer(self.conv1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2), ceil_mode = True)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            padding=(1,1),
            kernel_size=(3, 3),
            bias=False,
            stride=(1,1)



        )
        self.initialise_layer(self.conv2)

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            padding=(1,1),
            kernel_size=(3, 3),
            bias = False,
            stride=(1,1)


        )
        self.initialise_layer(self.conv3)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            padding=(1,1),
            kernel_size=(3, 3),
            bias= False,
            stride=(2,2),

        )
        self.initialise_layer(self.conv4)
        fcsize = 15488

        if(isMLMC):
           fcsize = 26048

        self.fc1 = nn.Linear(fcsize, 1024)
        self.initialise_layer(self.fc1)

        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

        #  self.smax = nn.Softmax(dim = 1)








    def forward(self, sounds: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.normaliseConv1(self.conv1(sounds)))

        x = F.relu(self.normaliseConv2(self.conv2(x)))
        x = self.dropout(x)

        x = self.pool1(x)


        x = F.relu(self.normaliseConv3(self.conv3(x)))
        x = F.relu(self.normaliseConv4(self.conv4(x)))
        x = self.dropout(x)

        x = torch.flatten(x, 1)


        x = torch.sigmoid((self.fc1(x)))
        x = self.dropout(x)




        x = self.fc2(x)

        #x = self.smax(x)



        return x

    @staticmethod
    def initialise_layer(layer):
        # if hasattr(layer, "bias"):
        #     nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
        #    if(type(layer) == nn.Linear):
        #        torch.nn.init.xavier_normal_(layer.weight, gain=1.),
        #    else:
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
        self.intermediate_results = None
        self.summary_writer = summary_writer

        self.step = 0
    # TODO
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 1,
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

                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()

                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    #preds = file_level_pred(preds)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)

                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            # self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                int_results = self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

        return int_results

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


    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        print("\n\nvalidating\n\n")
        results = {"preds": [], "labels": [], "fname": []}
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
                preds = logits.cpu().numpy()
                #preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))
                results["fname"].extend(list(fname))
            intermediate_results = copy.deepcopy(results)

            results = file_level_pred(results)

            print_accuracy(results, total_loss, self.val_loader)

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)


        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

        return intermediate_results


def print_accuracy(results, total_loss, val_loader):
    #accuracy = compute_accuracy(
    #    np.array(results["labels"]), np.array(results["preds"])
    #)
    compute_class_accuracy(
        np.array(results["labels"]), np.array(results["preds"])
    )
    if(total_loss):
        average_loss = total_loss / len(val_loader)
        print(f"validation loss: {average_loss:.5f}")


    #print(f"accuracy: {accuracy * 100:2.2f}")


def file_level_pred(results):
    file_results =  dict()
    for idx, prediction in enumerate(results["preds"]):
        file_name = results["fname"][idx]
        if(not file_name in file_results):
            #zarray = np.zeros(10)
            #zarray[np.argmax(prediction)] = 1
            file_results[file_name] = prediction
        else:
            #zarray = np.zeros(10)
            #zarray[np.argmax(prediction)] = 1
            file_results[file_name] = list(map(sum, zip(file_results[file_name], prediction)))

    for file_r in file_results:
        file_results[file_r] = np.argmax(file_results[file_r])

    for idx, result in enumerate(results["preds"]):
        results["preds"][idx] = file_results[results["fname"][idx]]


    return results

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:

    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    classLabel = [0] * 10
    classPred = [0] * 10
    acc = [0] * 10

    nameArray = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

    for i in range(len(labels)):
        if(labels[i] == preds[i]):
            classLabel[labels[i]] += 1
        classPred[labels[i]] += 1

    for i in range(10):
        if(classPred[i]):
            acc[i] = classLabel[i]/classPred[i]
        print(f"Class accuracy for {nameArray[i]}: {acc[i] * 100:2.2f}")

    totalAcc = 0
    for accur in acc:
        totalAcc += accur
    print(f"Overall Accuracy: {totalAcc * 10:2.2f}")

def run(mode):

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


    mode = mode
    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', mode),
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', mode),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)

    isMLMC = False
    if(mode == 'MLMC'):
        isMLMC = True


    model = CNN(height=41, width=85, channels=1, class_count=10, dropout=0.5, isMLMC=isMLMC)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    log_dir = get_summary_writer_log_dir()
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    int_results = trainer.train(
        epochs=50,
        print_frequency=50,
        val_frequency=1,
    )

    return int_results


def get_summary_writer_log_dir() -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bs=32_lr=0.001_run_'
    i = 0
    while i < 1000:
        tb_log_dir = Path("logs") / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)





if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("No mode given in arguments")
    else:
        mode = sys.argv[1]
        if(mode == 'TSCNN'):
            int_results1 = run('LMC')
            int_results2 = run('MC')
            combinedPreds = list(map(sum, zip(int_results1["preds"], int_results2["preds"])))
            int_results1["preds"] = combinedPreds
            results = file_level_pred(int_results1)
            print_accuracy(results, None, None)
            amountRight = np.equal(results["labels"], results["preds"])
            fileNameRight = mode + "right" + "modified"
            fileNameLabels = mode + "labels" + "modified"
            fileNamePreds = mode + "preds" + "modified"
            np.savetxt(fileNameRight, amountRight)
            np.savetxt(fileNameLabels, results["labels"])
            np.savetxt(fileNamePreds, results["preds"])
        elif(mode == 'LMC' or mode == 'MC' or mode == 'MLMC'):
            results = run(mode)
            results = file_level_pred(results)
            amountRight = np.equal(results["labels"], results["preds"])
            fileNameRight = mode + "right" + "modified"
            fileNameLabels = mode + "labels" + "modified"
            fileNamePreds = mode + "preds" + "modified"
            np.savetxt(fileNameRight, amountRight)
            np.savetxt(fileNameLabels, results["labels"])
            np.savetxt(fileNamePreds, results["preds"])

        else:
            print("Wrong arguments given")
