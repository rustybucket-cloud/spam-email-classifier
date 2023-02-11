import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features, out_features, h1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.out = nn.Linear(h1, out_features)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = torch.sigmoid(self.out(X))
        return X
    
def train(model, X_train, y_train, epochs=1):
    print("Beginning training\n\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs):
        if i % 10 == 0:
            print(f"Epoch: {i+1}")
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

def test(model, X_test, y_test):
    print("Beginning testing\n\n")
    criterion = nn.CrossEntropyLoss()
    correct = 0
    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval,y_test)
        for i, val in enumerate(y_eval):
            if y_test[i][0] == 1:
                if i % 50 == 0:
                    print(val, 'ham')
                if val[0] < 0.7:
                    correct += 1
            else:
                if val[0] > 0.7:
                    if i % 50 == 0:
                        print(val, 'spam')
                    correct += 1
    return f"""
    Total: {len(X_test)}
    Correct: {correct}
    Percentage: {correct / len(X_test)}
    Loss: {loss}
    """

