import torch
from torch import nn
import numpy as np

class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels:int, d_model:int, dim:int):
        # if the input labels are [1, 2, 3], I could concatenate the embeddings of 1, 2, 3 and then feed them into the model.
        assert d_model % 2 == 0
        super().__init__()
        self.num_embeddings_1 = nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0) # why +1?
        self.num_embeddings_2 = nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0)
        self.num_embeddings_3 = nn.Embedding(num_embeddings=num_labels, embedding_dim=d_model, padding_idx=0)
        self.condEmbedding = nn.Sequential(
            nn.Linear(d_model * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor: # x is the label rather than the t (time step).
        # emb_list = [self.num_embeddings(xi) for xi in x]
        emb_list = [self.num_embeddings_1(x[0]), self.num_embeddings_2(x[1]), self.num_embeddings_3(x[2])]
        concatenated_emb = torch.cat(emb_list, dim=-1)
        emb = self.condEmbedding(concatenated_emb)
        return emb
