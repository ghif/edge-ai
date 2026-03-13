from huggingface_hub import snapshot_download
import os

model_id = "google/medgemma-1.5-4b-it"
download_path = "./medgemma-1.5-4b-pytorch"

snapshot_download(
    repo_id=model_id,
    local_dir=download_path,
    token=os.getenv("HF_TOKEN")
)