import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, emb_size, output_size):
        super(RNN, self).__init__()

        self.learning_rate = 0.005
        self.criterion = nn.NLLLoss()
        self.hidden_size = hidden_size
        # self.emb_size = emb_size

        # self.i2e = nn.Linear(input_size, emb_size)  # Embedding layer
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # TODO: adapter éventuellement la dernière couche suivant la fonction loss que vous choisissez

    def forward(self, input, hidden):
        # embedding = torch.cat((input, hidden), 1)
        # combined = self.i2e(embedding)
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def train(self, emotion_tensor, line_tensor):
        # Create a zeroed initial hidden state
        hidden = self.init_hidden()
        self.zero_grad()

        # Read each letter in line tensore and keep hidden state for next letter
        for i in range(line_tensor.size()[0]):
            output, hidden = self(line_tensor[i], hidden)

        # Compare final output to target 
        loss = self.criterion(output, emotion_tensor)
        
        # Back-propagate
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in self.parameters():
            p.data.add_(p.grad.data, alpha=-self.learning_rate)

        # Return the ouptut and loss 
        return output, loss.item()