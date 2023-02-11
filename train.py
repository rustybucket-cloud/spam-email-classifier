import torch
from preprocess import get_sets, get_vectors, get_tokens
from model import Model, train, test

def main():
    X_train, X_test, y_train, y_test = get_sets()
    X_train, vectorizer = get_vectors(X_train)
    X_train = torch.Tensor(X_train.toarray())
    X_train.requires_grad = True

    model = Model(X_train.shape[1], 2, 100)
    y_train = torch.Tensor(y_train)
    y_train.requires_grad=True
    model = train(model, X_train, y_train, 50)

    documents = get_tokens(X_test)
    X_test = vectorizer.transform(documents)
    X_test = torch.Tensor(X_test.toarray())
    y_test = torch.Tensor(y_test)
    print(test(model, X_test, y_test))
    torch.save(model.state_dict(), f"./model.pt")

if __name__ == "__main__":
    main()