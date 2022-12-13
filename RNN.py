import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_dim, output_size, batch_size):
        super(RNN, self).__init__()

        self.learning_rate = 0.005
        self.batch_size = batch_size
        self.criterion = nn.NLLLoss()
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.input_size = input_size
        self.n_hidden_layer = 1

        self.emb = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size)
        self.rnn = nn.RNN(input_size=emb_size, hidden_size=hidden_dim, num_layers=self.n_hidden_layer, batch_first=True)
        self.lin = nn.Linear(hidden_dim, output_size)
        
        # self.i2e = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_size) 
        # self.i2h = nn.Linear(input_size, hidden_dim)
        # self.i2o = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # TODO: adapter éventuellement la dernière couche suivant la fonction loss que vous choisissez

    def forward(self, input, hidden):
        print(input.shape)
        print(hidden.shape)
        # embedding = torch.cat((input, hidden), 1)
        # combined = self.i2e(embedding)
        # combined = torch.cat((input, hidden), dim=1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # output = self.softmax(output)

        embeddings = self.emb(input)
        output, hidden = self.rnn(embeddings, torch.randn(self.n_hidden_layer, input.size(0), self.hidden_dim))
        output = self.lin(output[:,-1])
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, input):
        print("init hidden", input.shape)
        return torch.zeros(self.n_hidden_layer, self.batch_size, self.hidden_dim)

    def train(self, x, t):
        # Create a zeroed initial hidden state
        hidden = self.init_hidden(x)
        self.zero_grad()

        # Read each letter in line tensore and keep hidden state for next letter
        output, hidden = self(x, hidden)

        # Compare final output to target
        loss = self.criterion(output, t)

        # Back-propagate
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)

        # Return the ouptut and loss
        return output, loss.item()
