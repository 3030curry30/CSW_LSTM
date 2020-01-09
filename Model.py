import torch
import torch.nn as nn


class LSTM_Cell(nn.Module):
    def __init__(self, config, vocab):
        super(LSTM_Cell, self).__init__()
        self.vocab_size = len(vocab) + 10

        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.device = config.device

        self.hc = self.init_hidden(0)

        self.embed = nn.Embedding(self.vocab_size, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        return (torch.randn(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float).to(self.device),
                torch.randn(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float).to(self.device))

    def forward(self, word_ids):
        # init h0 and c0
        self.hc = self.init_hidden(word_ids.size(0))

        # get word embedding
        word_embedding = self.embed(word_ids)

        # get each token feature
        output, self.hc = self.lstm(word_embedding)
        return output


class GRU_Cell(nn.Module):
    def __init__(self, config, vocab):
        super(GRU_Cell, self).__init__()
        self.vocab_size = len(vocab) + 10

        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.device = config.device

        self.hc = self.init_hidden(0)

        self.embed = nn.Embedding(self.vocab_size, self.input_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def init_hidden(self, batch_size):
        return torch.randn(self.num_layers * 2, batch_size, self.hidden_size, dtype=torch.float).to(self.device)

    def forward(self, word_ids):
        # init h0
        self.hc = self.init_hidden(word_ids.size(0))

        # get word embedding
        word_embedding = self.embed(word_ids)

        # get each token feature
        output, self.hc = self.lstm(word_embedding)
        return output


class LSTM_Linear(nn.Module):
    def __init__(self, config, vocab):
        super(LSTM_Linear, self).__init__()

        self.lstm = LSTM_Cell(config, vocab)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        word_feature = self.lstm(x)
        output = self.fc(word_feature)
        return output