import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stopwords = set(stopwords.words('english'))

from sklearn.metrics import classification_report, confusion_matrix

import os
from tqdm import tqdm
tqdm.pandas()
from collections import Counter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_file_path = '/home/jovyan/Desktop/Wongyu/RNN/IMDB_Dataset/Train.csv'
test_file_path = '/home/jovyan/Desktop/Wongyu/RNN/IMDB_Dataset/Test.csv'
valid_file_path = '/home/jovyan/Desktop/Wongyu/RNN/IMDB_Dataset/Valid.csv'

df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)
df_valid = pd.read_csv(valid_file_path)

def simplize(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    tokens = word_tokenize(text)
    no_stopwords = [i for i in tokens if i not in stopwords]
    lemmas = [WordNetLemmatizer().lemmatize(t) for t in no_stopwords]
    return ' '.join(lemmas)

df_train['eng_review'] = df_train.text.progress_apply(simplize)
df_test['eng_review'] = df_test.text.progress_apply(simplize)
df_valid['eng_review'] = df_valid.text.progress_apply(simplize)

def word(text):
    return word_tokenize(text)

df_train['word'] = df_train.eng_review.progress_apply(word)
df_test['word'] = df_test.eng_review.progress_apply(word)
df_valid['word'] = df_valid.eng_review.progress_apply(word)

from collections import Counter

all_words = [word for df in [df_train, df_valid, df_test] for sublist in df['word'] for word in sublist]

counter = Counter(all_words)
vocab = sorted(counter, key=counter.get, reverse=True)
int2word = dict(enumerate(vocab, 1))
int2word[0] = '<PAD>'

word2index = {word: idx for idx, word in enumerate(int2word, 1)}
word2index['<PAD>'] = 0

def text_to_int_sequence(text, word2index):
    return [word2index.get(word, 0) for word in text.split()]

def pad_sequence(sequence, max_length):
    return sequence + [0] * (max_length - len(sequence))

df_train['encoded_text'] = df_train['eng_review'].apply(lambda x: text_to_int_sequence(x, word2index))
df_valid['encoded_text'] = df_valid['eng_review'].apply(lambda x: text_to_int_sequence(x, word2index))
df_test['encoded_text'] = df_test['eng_review'].apply(lambda x: text_to_int_sequence(x, word2index))

max_seq_length = max(df_train['encoded_text'].apply(len).max(), df_valid['encoded_text'].apply(len).max())

df_train['padded_encoded_text'] = df_train['encoded_text'].apply(lambda x: pad_sequence(x, max_seq_length))
df_valid['padded_encoded_text'] = df_valid['encoded_text'].apply(lambda x: pad_sequence(x, max_seq_length))
df_test['padded_encoded_text'] = df_test['encoded_text'].apply(lambda x: pad_sequence(x, max_seq_length))

batch_size = 32
trainset = TensorDataset(torch.Tensor(df_train['padded_encoded_text'].tolist()), torch.Tensor(df_train['label'].tolist()))
validset = TensorDataset(torch.Tensor(df_valid['padded_encoded_text'].tolist()), torch.Tensor(df_valid['label'].tolist()))
testset = TensorDataset(torch.Tensor(df_test['padded_encoded_text'].tolist()), torch.Tensor(df_test['label'].tolist()))

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, drop_last=True)
valloader = DataLoader(validset, shuffle=True, batch_size=batch_size, drop_last=True)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size, drop_last=True)


class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, hidden):
        h_t = torch.tanh(self.W_ih @ x.t() + self.b_ih + self.W_hh @ hidden + self.b_hh)
        return h_t

class CustomRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, embedding_size):
        super(CustomRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.custom_rnn_cell = CustomRNNCell(input_size=embedding_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)

        hidden = torch.zeros(x.size(0), self.custom_rnn_cell.hidden_size, device=x.device)

        for t in range(x.size(1)):
            hidden = self.custom_rnn_cell.forward(x[:, t, :], hidden)

        out = self.fc(hidden)
        return out


vocab_size = len(vocab)
output_size = 1
hidden_size = 32
embedding_size = 400
RNN_model = CustomRNN(vocab_size, output_size, hidden_size, embedding_size)
model = RNN_model.to(device)

lr = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(RNN_model.parameters(), lr=lr)
grad_clip = 5
epochs = 10
print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'Test_loss': [],
    'Test_acc': [],
    'epochs': epochs
}

epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

for e in epochloop:
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for id, (feature, target) in enumerate(trainloader):
        epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')
        feature, target = feature.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(feature)
        out = torch.sigmoid(out)
        target = target.unsqueeze(1)
        loss = criterion(out, target)
        train_loss += loss.item()
        predicted = (out > 0.5).float()
        correct_train += (predicted == target).sum().item()
        total_train += target.size(0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    train_accuracy = correct_train / total_train
    history['train_loss'].append(train_loss / len(trainloader))
    history['train_acc'].append(train_accuracy)

    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for id, (feature, target) in enumerate(valloader):
            epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            out = torch.sigmoid(out)
            target = target.unsqueeze(1)
            loss = criterion(out, target)
            val_loss += loss.item()
            predicted = (out > 0.5).float()
            correct_val += (predicted == target).sum().item()
            total_val += target.size(0)

    val_accuracy = correct_val / total_val
    history['val_loss'].append(val_loss / len(valloader))
    history['val_acc'].append(val_accuracy)

    model.eval()
    test_loss = 0
    correct_Test = 0
    total_Test = 0
    with torch.no_grad():
        for id, (feature, target) in enumerate(testloader):
            epochloop.set_postfix_str(f'Test batch {id}/{len(testloader)}')
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            out = torch.sigmoid(out)
            target = target.unsqueeze(1)
            loss = criterion(out, target)
            test_loss += loss.item()
            predicted = (out > 0.5).float()
            correct_Test += (predicted == target).sum().item()
            total_Test += target.size(0)

    Test_accuracy = correct_Test / total_Test
    history['Test_loss'].append(test_loss / len(testloader))
    history['Test_acc'].append(Test_accuracy)

    epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(trainloader):.6f} | Train Acc: {train_accuracy:.6f} | Val Loss: {val_loss / len(valloader):.6f} | Val Acc: {val_accuracy:.6f} | Test Loss: {test_loss / len(testloader):.6f} | Test Acc: {Test_accuracy:.6f}')
    epochloop.update()

print(history)
