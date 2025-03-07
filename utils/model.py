import torch
import torch.nn as nn

class ToxicClassifier(nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, num_filters, kernel_size, dropout, num_classes):
        super().__init__()
        # Embedding layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        # CNN layer
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1)
        self.pool = nn.AdaptiveMaxPool1d(50) # This reduces the sequence length
        # GRU layer
        self.gru = nn.GRU(
            input_size=num_filters,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True)
        # Fully connected layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1) # change shape for conv1d (batch_size, channels, seq_len)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1) # change shape back for GRU (batch_size, seq_len, channels)
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :]) # take the last time step
        return self.fc(x)
