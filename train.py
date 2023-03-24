import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ChatBot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatBot, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = input.unsqueeze(0)
        embedded = self.encoder(input)
        output, hidden = self.gru(embedded, self.hidden)
        self.hidden = hidden
        output = self.decoder(output.squeeze(0))
        output = self.softmax(output)
        return output

    def init_hidden(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)

class ChatBotDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.data.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_string, target_string = self.data[idx]
        input_tensor = self.tensor_from_input(input_string)
        target_tensor = self.tensor_from_output(target_string)
        return input_tensor, target_tensor

    def tensor_from_input(self, input_string):
        input_indices = [int(index) for index in input_string.split()]
        tensor = torch.LongTensor(input_indices)
        return tensor

    def tensor_from_output(self, output_string):
        output_indices = [int(index) for index in output_string.split()]
        output_indices.append(2) # append <EOS> token
        tensor = torch.LongTensor(output_indices)
        return tensor

class Trainer:
    def __init__(self, dataset_path):
        self.dataset = ChatBotDataset(dataset_path)
        self.input_size = None
        self.output_size = None
        self.word2index = None
        self.index2word = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def train(self, n_epochs=1000, learning_rate=0.01, save_path=None):
        self.build_vocabulary()
        self.build_model()
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        train_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        for epoch in range(n_epochs):
            for input_tensor, target_tensor in train_loader:
                self.train_step(input_tensor.squeeze(0), target_tensor.squeeze(0))
        if save_path is not None:
            self.save_model(save_path)

    def build_vocabulary(self):
        words = set()
        for input_string, target_string in self.dataset.data:
            words.update(input_string.split())
            words.update(target_string.split())
        words = list(set(words))
        self.input_size = len(words)
        self.output_size = len(words)
        self.word2index = {word: i for i, word in enumerate(words)}
        self.index2word = {i: word for i, word in enumerate(words)}

    def build_model(self, hidden_size=128):
        self.model = ChatBot(self.input_size, hidden_size, self.output_size)

    def tensor_from_input(self, input_string):
        input_indices = [self.word2index[word] for word in input_string.split()]
        tensor = torch.LongTensor(input_indices)
        return tensor

    def tensor_from_output(self, output_string):
        output_indices = [self.word2index[word] for word in output_string.split()]
        output_indices.append(self.word2index["<EOS>"])
        tensor = torch.LongTensor(output_indices)
        return tensor

    def train_step(self, input_tensor, target_tensor):
        self.model.init_hidden()
        self.optimizer.zero_grad()
        loss = 0
        target_length = target_tensor.size(0)
        for i in range(input_tensor.size(0)):
            output = self.model(input_tensor[i])
        for i in range(target_length):
            output = self.model(target_tensor[i])
            loss += self.criterion(output, target_tensor[i].unsqueeze(0))
        loss.backward()
        self.optimizer
        
# Initialize a new Trainer object and specify the path to the training data
# need to download the Cornell Movie Dialogs Corpus from http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
trainer = Trainer("data/data.txt")

# Train the chatbot for 1000 epochs and save the model to a file called "model.pth"
trainer.train(n_epochs=1000, learning_rate=0.01, save_path="model.pth")