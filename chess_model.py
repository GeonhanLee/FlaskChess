import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self, num_emb, output_size, num_layers=1, hidden_size=128):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create an embedding layer to convert token indices to dense vectors
        self.embedding = nn.Embedding(num_emb, hidden_size)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.5)

        # Define the output fully connected layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        # Convert token indices to dense vectors
        input_embs = self.embedding(input_seq)

        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size, device=input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size, device=input_seq.device)

        # Pass the embeddings through the LSTM layer
        output, _ = self.lstm(input_embs, (h0, c0))

        # Pass the LSTM output through the fully connected layer to get the final output
        output = self.fc_out(output[:, -1, :])  # out: tensor of shape (batch_size, num_classes)
        return output