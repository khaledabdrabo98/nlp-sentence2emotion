import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_dim, output_size, batch_size, lr):
        super(RNN, self).__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size

        self.i2e = torch.nn.Linear(input_size, emb_size)
        self.i2h = torch.nn.Linear(emb_size + hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.i2o = torch.nn.Linear(emb_size + hidden_dim, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # print(f"input: {input.shape}, hidden: {hidden.shape}")
        lin_input = self.i2e(input)
        combined = torch.cat((lin_input, hidden), 1)
        hidden = self.relu(combined)
        hidden = self.i2h(hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.int64)
