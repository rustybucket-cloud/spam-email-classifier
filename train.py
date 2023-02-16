import torch
from preprocess import get_sets, get_vectors, get_tokens
from model import Model, train, test
import pickle
from traininglog import log_about

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
            if confidence > 0.5:
                confidence = round((confidence - 0.5) * 2, 2)
                return { "value": "Spam", "confidence": confidence }
            else:
                confidence = round((0.5 - confidence) * 2, 2)
                return { "value": "Not Spam", "confidence": confidence }
    

def main():
    train_data, test_data, features_len = get_sets()
    print(f"Train set size: {len(train_data)}\tTest set size: {len(test_data)}")

    hidden_layer_nodes = 200
    model = Model(features_len, 1, hidden_layer_nodes)
    if torch.cuda.is_available():
        print("cuda available\n")
        model.cuda()
        train.to(device=torch.device('cuda:0'))
        test.to(device=torch.device('cuda:0'))

    max_epochs = 100
    lr = 0.01
    model, loss = train(model, train_data, features_len, max_epochs=max_epochs, lr=lr, loss_goal=0.25)

    result, percentage = test(model, test_data, features_len)
    print(result)
    torch.save(model.state_dict(), f"./model_2.pt")

    predictor = SpamModel(model, get_tokens)
    
    with open("./model_2.pkl", "wb") as f:
        pickle.dump(predictor, f)
    
    log_about(len(train_data), len(test_data), max_epochs, lr, loss, percentage, hidden_layer_nodes)


if __name__ == "__main__":
    main()
    # # with open("./model.pkl", "rb") as f:
    #     # model = pickle.load(f)
    # train_data, test_data, features_len = get_sets()
    # model = Model(features_len, 1, 100)
    # model.load_state_dict(torch.load("./model.pt"))
    # print(test(model, test_data, features_len))