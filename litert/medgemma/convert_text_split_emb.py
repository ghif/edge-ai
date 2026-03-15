import os
import torch
import litert_torch as ai_edge_torch
from litert_torch.generative.layers import model_config as cfg
from litert_torch.generative.utilities import model_builder
from litert_torch.generative.utilities import loader as loading_utils
from litert_torch.generative.examples.gemma3 import decoder

class DecomposedEmbedding(torch.nn.Module):
    def __init__(self, original_emb):
        super().__init__()
        self.original_emb = original_emb
        self.vocab_size = original_emb.num_embeddings
        self.dim = original_emb.embedding_dim
        # Split into 2 halves
        self.half = self.vocab_size // 2
        
    def forward(self, x):
        # This is a hack to see if it bypasses the "illegal" op
        # We use a mask to split the lookup
        mask = (x < self.half).long()
        x1 = x * mask
        x2 = (x - self.half) * (1 - mask)
        # Still calls embedding, but maybe the smaller table helps?
        # No, that won't work because both halves are in the same table.
        
        # Real decomposition:
        # Actually, let's just try to use torch.gather
        return torch.index_select(self.original_emb.weight, 0, x.view(-1)).view(x.shape + (self.dim,))

def get_medgemma_config():
    # ... (same as before)
    pass

# I'll just use the standard Decoder but I'll try to wrap it if possible.
# But Decoder is a complex module.

# Wait! I'll try to use 'dynamic_int8' but I'll exclude ONLY the embedding.
# That was in the original script!

print("Attempting conversion with dynamic_int8 and excluded embedding...")
# ...
