import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, config, vocab_size, pad_id, pretrained_embedding=None, freeze_embedding=False):
        super().__init__()
        self.config = config
        self.emb_dim = self.config.emb_dim
        self.dropout = self.config.dropout
        self.num_classes = self.config.num_classes
        self.num_filters = self.config.num_filters
        self.filter_sizes = self.config.filter_sizes

        if pretrained_embedding is not None:
            self.vocab_size, self.emb_dim = pretrained_embedding.size()
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)

        else:
            self.embedding = nn.Embedding(vocab_size, self.emb_dim, padding_idx=pad_id)

        self.cnn_layers = nn.ModuleList([nn.Conv1d(in_channels=self.emb_dim, out_channels=n, kernel_size=s) for s, n in zip(self.filter_sizes, self.num_filters)])
        self.fc = nn.Linear(sum(self.num_filters), self.num_classes)
        self.dropout = nn.Dropout(self.dropout)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, inputs):

        inputs_embed = self.embedding(inputs).float()

        # inputs dim : (b, max_len, emb_dim)
        # Permute inputs dimension to match nn.Conv1d : (b, emb_dim, max_len)
        inputs_reshaped = inputs_embed.permute(0, 2, 1)

        # Apply CNN and ReLU : (b, num_filter, L_out)
        outputs = [F.relu(conv1d(inputs_reshaped)) for conv1d in self.cnn_layers]

        # Apply Max Pooling : (b, emb_dim, 1)
        outputs = [F.max_pool1d(output, kernel_size=output.shape[2]) for output in outputs]

        # Concatenate
        concat_outputs = torch.cat([output.squeeze(dim=-1) for output in outputs], dim=1)

        # Apply Fully connected layer
        logits = self.fc(self.dropout(concat_outputs))

        return logits

if __name__ == "__main__":
    m = nn.Conv1d(16,33,3)
    input = torch.randn(20, 16, 50)
    output = m(input)
    print(m(input).size())

    print(F.max_pool1d(output, kernel_size=output.shape[2]).shape)

    
    