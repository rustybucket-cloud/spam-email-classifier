import torch
from preprocess import get_sets, get_vectors, get_tokens
from model import Model, train, test
import pickle

class SpamModel:
    def __init__(self, model, vectorizer, get_tokens):
        self.model = model
        self.vectorizer = vectorizer
        self.get_tokens = get_tokens

    def classify(self, document):
        X = self.get_tokens([document])
        X = self.vectorizer.transform(X)
        with torch.no_grad():
            pred = self.model.forward(torch.Tensor(X.toarray()))
            if pred[0][0] > pred[0][1]:
                return "Not Spam"
            else:
                return "Spam"
    

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

    predictor = SpamModel(model, vectorizer, get_tokens)
    
    with open("./model.pkl", "wb") as f:
        pickle.dump(predictor, f)


if __name__ == "__main__":
    main()