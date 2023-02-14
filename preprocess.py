import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import os
import math
import pickle
import torch
from torch.utils.data import TensorDataset, Subset

stemmer = SnowballStemmer("english")

def tokenize(document):
    try:
        tokens = word_tokenize(document)
    except:
        print("Error tokenizing document")
        return []
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def create_vector_list(documents):
    vector_list = []
    for document in documents:
        tokens = tokenize(document)
        for token in tokens:
            if len(token) < 30 and re.search("\w", token) != None and re.search("\W", token) == None and token not in vector_list:
                vector_list.append(token)
    return vector_list

def create_vector_dicts(vector_list):
    vector_dict = {token: i for i, token in enumerate(vector_list)}
    reverse_vector_dict = {token: i for i, token in enumerate(vector_list)}
    return vector_dict, reverse_vector_dict

def get_tokens(items):
    documents = []
    for item in items:
        tokens = tokenize(item)
        documents.append(" ".join(tokens))
    return documents

def get_vectors(items):
    vectorizer = TfidfVectorizer()
    # vectorizer = CountVectorizer()
    documents = get_tokens(items)
    X = vectorizer.fit_transform(documents)
    return X, vectorizer

def get_sets():
    print("Getting sets\n")
    df = pd.read_csv("./emails.csv")
    X = df["Text"].values

    TEST_SIZE = 0.1
    SEED = 42

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, test_indices, _, _ = train_test_split(
        range(len(X)),
        X,
        # stratify=X.targets,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    print("Getting vectors\n")
    if os.path.exists("./vectorizer.pkl"):
        with open("./vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        X = vectorizer.transform(X)
    else:
        X, vectorizer = get_vectors(X)
        with open("./vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    y = [[1] if val == "spam" else [0] for val in df["Type"].values]
    dataset = TensorDataset(torch.IntTensor(df.index.values), torch.Tensor(y))
    item_len = len(X.toarray())
    train_size = math.floor(item_len / .75)

    print("Creating training and testing groups\n")
    # generate subset based on indices
    train = Subset(dataset, train_indices)
    test = Subset(dataset, test_indices)
    # train, test = torch.utils.data.random_split(dataset, [train_size, item_len - train_size], generator=torch.Generator().manual_seed(42))
    return train, test, len(vectorizer.vocabulary_)

def vectorize(documents, vector_dict):
    vectors = np.zeros([len(documents), len(vector_dict.keys())])
    for i, document in enumerate(documents):
        tokens = tokenize(document)
        for token in tokens:
            if token in vector_dict:
                vectors[i][vector_dict[token]] += 1
    return vectors

def process_email(email):
    lines = email.splitlines()
    lines = [line for line in lines if (re.search("\w", line) != None and re.search("(\s*<?\w+:.*)|(jmason.org)|(fetchmail-5.9.0)|(jm@jmason.org)|(jm@jmason.org)|(fork@xent.com)|(___)|(\w+\.\w+\.\w+)|(<\S+>)|((\+|-)\d{4})|(tests=)|(version=)|(Postfix)|(([A-Z]|_)+)|(- - - - -)", line) == None)]
    return "\t".join(lines)

def walk_dir(dir, items, title):
    for dirname, dirs, filenames in os.walk(dir):
        for i, filename in enumerate(filenames):
            if i % 50 == 0:
                print(f"Item {i + 1} of {len(filenames)}")
            try:
                with open(dir + "/" + filename, "r") as f:
                    text = f.read()
                    text = process_email(text)
                    if text != "":
                        items.append([title, text])
            except:
                continue
        if len(dirs) > 0:
            for item in dirs:
                walk_dir(dir + "/" + item, items, title)
    return items

def create_sets():
    emails = []
    walk_dir("./emails/ham", emails, "ham")
    walk_dir("./emails/spam", emails, "spam")
    df = pd.DataFrame(emails, columns=["Type", "Text"])
    df.to_csv("./emails_2.csv")

if __name__ == "__main__":
    create_sets()
