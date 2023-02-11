import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
    documents = get_tokens(items)
    X = vectorizer.fit_transform(documents)
    return X, vectorizer

def get_sets():
    df = pd.read_csv("./emails.csv")
    X = df["text"].values
    y = [[1,0] if val == "spam" else [0,1] for val in df["type"].values]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def vectorize(documents, vector_dict):
    vectors = np.zeros([len(documents), len(vector_dict.keys())])
    for i, document in enumerate(documents):
        tokens = tokenize(document)
        for token in tokens:
            if token in vector_dict:
                vectors[i][vector_dict[token]] += 1
    return vectors