# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

# sklearn
from sklearn.metrics import classification_report, confusion_matrix

# utils
import os
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

data = pd.read_csv('/content/drive/MyDrive/AI/pytorch/archive/IMDB Dataset.csv')
data.head()

def transform_label(label):
    return 1 if label == 'positive' else 0

data['label'] = data['sentiment'].progress_apply(transform_label)
data.head()

from sklearn.model_selection import train_test_split

X,y = data['review'].values,data['sentiment'].values
x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
print(f'shape of train data is {x_train.shape}')
print(f'shape of test data is {x_test.shape}')


import re
import nltk
nltk.download('stopwords')

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(nltk.corpus.stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                    if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = [1 if label =='positive' else 0 for label in y_train]
    encoded_test = [1 if label =='positive' else 0 for label in y_val]
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)

x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_p, output_size, model_type='LSTM'):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'LSTM':
            self.sequenceclassifier = nn.LSTM(
                input_size = embedding_dim,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout_p
            )
        elif model_type == 'GRU':
            self.sequenceclassifier = nn.GRU(
                input_size = embedding_dim,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True,
                dropout = dropout_p
            )
        self.fc = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        output, _ = self.sequenceclassifier(embeds) # |output| = (128, 10, 32)
        output = output[:, -1, :] # |output| = (128, 32)
        y = self.fc(output)
        return y

  def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def train(model, loader, criterion, optimizer, device, config, wandb):
    model.train()
    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(config.epochs):
        cumu_loss = 0
        for inputs, labels in loader:
            inputs, labels  = inputs.to(device), labels.to(device)
            # print("labels")
            # print(labels)

            output = model(inputs)

            # print("output")
            # print(output)

            loss = criterion(output.squeeze(), labels.float())
            cumu_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = cumu_loss / len(loader)
        wandb.log({"train_loss": avg_loss}, step=epoch)
        print(f"TRAIN: EPOCH {epoch + 1:04d} / {config.epochs:04d} | Epoch LOSS {avg_loss:.4f}")


def vaild(model, loader, criterion, device,  wandb):
    model.eval()

    with torch.no_grad():
        correct, test_loss = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


    val_loss = test_loss / len(loader)
    print(f"VALID: LOSS {val_loss:.4f} | Accuracy {val_loss:.4f} ")
    wandb.log({
        "val_acc": 100. * correct / len(loader.dataset),
        "val_loss": val_loss})
