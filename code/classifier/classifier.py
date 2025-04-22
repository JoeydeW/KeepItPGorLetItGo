import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import ToTensor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tqdm import tqdm

class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.l2 = nn.Linear(16, 4)
        self.l3 = nn.Linear(4, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super(Dataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = np.array(self.data[idx], dtype=np.float32)
        # label = np.zeros(4, dtype=np.float32)
        # label[self.labels[idx]] = 1.0
        return sample, self.labels[idx]

def load_dataset(filepath):
    with open(filepath, "r+", encoding="UTF-8") as file:
        data = json.loads(file.read())
    return data

def prepare_data(dataset, labelset, batch_size):
    train_data, test_data = dataset
    train_labels, test_labels = labelset
    train_dataloader = DataLoader(dataset=Dataset(train_data, train_labels), batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=Dataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def initialize_network(input_size, output_size):
    classifier = Classifier(input_size, output_size).to("cuda")
    return classifier

def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    history = []

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            data, target = inputs.to("cuda"), labels.to("cuda")
            scores = model(data)
            loss = criterion(scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history.append(running_loss / len(train_loader))

    return model, history

# def test(model, test_loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, labels in test_loader:
#             data, target = data.to("cuda"), labels.to("cuda")
#             scores = model(data)
#             _, predictions = scores.max(1)
#             print(predictions, target)
#             correct += (predictions.eq(target).sum()).item()
#             # correct += predictions.eq(torch.argmax(target, dim=1)).sum()
#             total += predictions.size(0)
#
#     print(f"Accuracy: {100 * correct / total:.2f}%")

def test(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, target = data.to("cuda"), labels.to("cuda")
            scores = model(data)
            _, predictions = scores.max(1)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)

    accuracy = report["accuracy"]
    precision = report["macro avg"]["precision"]
    recall = report["macro avg"]["recall"]
    f1 = report["macro avg"]["f1-score"]

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    INPUT_SIZE = 75
    OUTPUT_SIZE = 4
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 8
    EPOCHS = 100

    data = load_dataset("../data/stratified_folds/stratified_folds_12_features.json")
    labels = load_dataset("../data/stratified_folds/stratified_folds_labels_12_features.json")

    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for fold_idx, (d, l) in enumerate(zip(data, labels), 1):
        print(f"\n=== Fold {fold_idx} ===")
        training_loader, test_loader = prepare_data(d, l, BATCH_SIZE)

        model = initialize_network(INPUT_SIZE, OUTPUT_SIZE)
        model, history = train(model, training_loader, EPOCHS, LEARNING_RATE)

        acc, prec, rec, f1 = test(model, test_loader)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        plt.plot(history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve (Fold {fold_idx})")
        plt.grid()
        plt.show()

    print("\n=== 8-Fold Cross-Validation Summary ===")
    print(f"Average Accuracy : {np.mean(accuracies) * 100:.2f}%")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall   : {np.mean(recalls):.4f}")
    print(f"Average F1-score : {np.mean(f1s):.4f}")