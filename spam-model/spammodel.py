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