import torch
import litert_torch as ai_edge_torch
import os

class SimpleEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(262208, 2560)
    def forward(self, x):
        return self.emb(x)

model = SimpleEmbedding().eval()
x = torch.zeros((1, 1), dtype=torch.long)

print("Converting simple embedding...")
try:
    edge_model = ai_edge_torch.convert(model, (x,))
    edge_model.export("test_emb.tflite")
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
