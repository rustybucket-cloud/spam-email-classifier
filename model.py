import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import statistics
from preprocess import get_tokens

class Model(nn.Module):
    def __init__(self, in_features, out_features, h1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.out = nn.Linear(h1, out_features)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X = torch.sigmoid(self.out(X))
        return X
    
def train(model, dataset, features_len, epochs=1):
    print("Beginning training\n")
    df = pd.read_csv("./emails.csv")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

    with open("./vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    losses = []

    for epoch in range(epochs):
        set_losses = []
        dataset_count = 1
        for X, y in dataloader:
            texts = torch.zeros(len(X), features_len)
            for i, val in enumerate(X):
                text = df.iloc[val.item()]["Text"]
                text = get_tokens([text])
                text = vectorizer.transform(text)
                text = text.toarray()
                texts[i] = torch.Tensor(text)

            y_pred = model.forward(texts)
            loss = criterion(y_pred, y)
            if dataset_count % 50 == 0:
                print(f"Epoch {epoch + 1}\tDataset: {dataset_count}")
            dataset_count += 1
            optimizer.zero_grad()
            loss.backward()

            set_losses.append(loss.item())
        losses.append(statistics.mean(set_losses))

        print(f"Epoch: {epoch + 1}\tLoss: {statistics.mean(set_losses)}")
        optimizer.step()
    
    plt.plot([n+1 for n in range(len(losses))], losses)
    plt.show()

    return model

def test(model, dataset, features_len):
    print("Beginning testing\n\n")
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=50)
    df = pd.read_csv("./emails.csv")

    with open("./vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    correct = 0
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            texts = torch.zeros(len(X), features_len)
            for i, val in enumerate(X):
                text = df.iloc[val.item()]["Text"]
                text = get_tokens([text])
                text = vectorizer.transform(text)
                text = text.toarray()
                texts[i] = torch.tensor(text)
            y_eval = model.forward(texts)
            loss = criterion(y_eval,y)
            losses.append(loss.item())
            for i, val in enumerate(y_eval):
                if i % 25 == 0:
                    print(f"Predited Value: {round(val.item(), 2)}\tTrue Value: {y[i].item()}")
                if int(y[i].item()) == 1:
                    if val.item() >= 0.5:
                        correct += 1
                else:
                    if val.item() < 0.5:
                        correct += 1
    return f"""
    Total: {len(dataset)}
    Correct: {correct}
    Percentage: {round(correct / len(dataset), 2) * 100}%
    Loss: {statistics.mean(losses)}
    """

