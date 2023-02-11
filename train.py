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
            confidence = pred.item()
            if confidence > 0.7:
                confidence = round(1 - ((1 - confidence)/ 0.3), 2)
            else:
                confidence = round(1 - (confidence / 0.7), 2)
            if pred[0] > .7:
                return { "value": "Spam", "confidence": confidence }
            else:
                return { "value": "Not Spam", "confidence": confidence }
    

def main():
    X_train, X_test, y_train, y_test = get_sets()
    X_train, vectorizer = get_vectors(X_train)
    X_train = torch.Tensor(X_train.toarray())
    X_train.requires_grad = True

    model = Model(X_train.shape[1], 1, 100)
    y_train = torch.Tensor(y_train)
    if torch.cuda.is_available():
        print("cuda available\n\n")
        model.cuda()
        X_train.to(device=torch.device('cuda:0'))
        y_train.to(device=torch.device('cuda:0'))
    y_train.requires_grad=True
    model = train(model, X_train, y_train, 50)

    documents = get_tokens(X_test)
    X_test = vectorizer.transform(documents)
    X_test = torch.Tensor(X_test.toarray())
    y_test = torch.Tensor(y_test)
    if torch.cuda.is_available():
        model.cuda()
        X_test.to(device=torch.device('cuda:0'))
        y_test.to(device=torch.device('cuda:0'))

    print(test(model, X_test, y_test))
    torch.save(model.state_dict(), f"./model.pt")

    predictor = SpamModel(model, vectorizer, get_tokens)
    
    with open("./model.pkl", "wb") as f:
        pickle.dump(predictor, f)


if __name__ == "__main__":
    main()