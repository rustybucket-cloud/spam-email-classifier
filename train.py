import torch
from preprocess import get_sets, get_vectors, get_tokens
from model import Model, train, test
import pickle

class SpamModel:
    def __init__(self, model, get_tokens):
        self.model = model
        with open("./vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
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
    train_data, test_data, features_len = get_sets()
    print(f"Train set size: {len(train_data)}\tTest set size: {len(test_data)}")

    model = Model(features_len, 1, 100)
    if torch.cuda.is_available():
        print("cuda available\n")
        model.cuda()
        train.to(device=torch.device('cuda:0'))
        test.to(device=torch.device('cuda:0'))

    model = train(model, train_data, features_len, epochs=100)

    print(test(model, test_data, features_len))
    torch.save(model.state_dict(), f"./model.pt")

    predictor = SpamModel(model, get_tokens)
    
    with open("./model.pkl", "wb") as f:
        pickle.dump(predictor, f)


if __name__ == "__main__":
    main()
    # # with open("./model.pkl", "rb") as f:
    #     # model = pickle.load(f)
    # train_data, test_data, features_len = get_sets()
    # model = Model(features_len, 1, 100)
    # model.load_state_dict(torch.load("./model.pt"))
    # print(test(model, test_data, features_len))