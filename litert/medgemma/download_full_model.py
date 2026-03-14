import os
import hf_transfer
from huggingface_hub import snapshot_download

# Enable hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

model_id = "google/medgemma-1.5-4b-it"
local_dir = "medgemma-1.5-4b-pytorch"
token = os.getenv("HF_TOKEN")

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    repo_type="model",
    token=token
)
print("Download complete.")
