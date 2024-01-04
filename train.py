
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
from torch import nn, optim
from torch.optim import Optimizer
import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from misc import load_data
    
data_path = 'datasets.json'
dataset = load_data(data_path)
labels_path = 'labels.json'
class_labels = load_data(labels_path)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        key: str =  "val",
    ):
        self.model.train()

        all_logits = []

        for epoch in range(start_epoch, epochs):
            self.model.train()

            data_load_start_time = time.time()
            for _, samples, label in self.train_loader:
                batch = samples.to(self.device)
                labels = label.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)
                all_logits.extend(logits.detach().cpu())
                loss = self.criterion(logits, labels)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
                
            self.scheduler.step(accuracy)

            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if ((epoch + 1) % val_frequency) == 0:

                self.validate(self.val_loader, dataset, key , "Validation: ")

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

    def validate(self, data_loader, dataset, dataset_key, mode):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()
        test_path = dataset[dataset_key]["annotations_path"]

        with torch.no_grad():

            for _, samples, label in data_loader:

                batch = samples.to(self.device)
                labels = label.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = np.mean(evaluate(results["preds"], test_path, mode))
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
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def compute_accuracy(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """
    Compute the average AUC-ROC score across all labels.
    
    Args:
        labels: (batch_size, class_count) tensor containing true labels.
        preds: (batch_size, class_count) tensor containing model predictions.

    Returns:
        avg_auc_score: Average AUC score across all labels.
    """
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()

    auc_scores = []
    for i in range(labels_np.shape[1]): 
        try:
            auc = roc_auc_score(labels_np[:, i], preds_np[:, i])
            auc_scores.append(auc)
        except ValueError:
            pass

    if auc_scores:  
        avg_auc_score = np.mean(auc_scores)
    else:
        avg_auc_score = 0.0  

    return avg_auc_score


def evaluate(preds, gts_path, additional_info):
    """
    Given the list of all model outputs (logits), and the path to the ground
    truth (val.pkl), calculate the AUC Score of the classified segments.
    Args:
        preds (List[torch.Tensor]): The model ouputs (logits). This is a
            list of all the tensors produced by the model for all samples in
            val.pkl. It should be a list of length 4332 (size of val). All
            tensors in the list should be of size 50 (number of classes).
        gts_path (str): The path to val.pkl
    Returns:
        auc_score (float): A float representing the AUC Score
    """
    #gts = torch.load(gts_path, map_location='cpu') # Ground truth labels, pass path to val.pkl
    gts = pd.read_pickle(gts_path)

    labels = []
    model_outs = []
    for i in range(len(preds)):
        # labels.append(gts[i][2].numpy())                             # A 50D Ground Truth binary vector
        labels.append(np.array(gts.iloc[i]['label']).astype(float))    # A 50D Ground Truth binary vector
        model_outs.append(preds[i].cpu().numpy()) # A 50D vector that assigns probability to each class

    labels = np.array(labels).astype(float)
    model_outs = np.array(model_outs)

    auc_score = roc_auc_score(y_true=labels, y_score=model_outs, average=None)
    avg_score = np.mean(auc_score)

    class_labels_dict = class_labels['class_labels_dict']
    
    print("EVALUATION METRICS:")
    print("-------------------------------------------------------------")
    print()

    if additional_info == "Validation: ":
        print(f'Validation AUC Score: {(avg_score):.4f}')

    elif additional_info == "Test: ":
        if len(class_labels_dict) == len(auc_score):
            label_auc_mapping = ', '.join([f'{class_labels_dict[str(i)]}: {auc_score[i]:.4f}' for i in range(len(auc_score))])
            print(f'Test AUC Score: {avg_score:.4f}, Class Labels: {label_auc_mapping}')
        else:
            print("Error: The number of class labels and AUC scores do not match.")

    print()
    print("-------------------------------------------------------------")

    return auc_score