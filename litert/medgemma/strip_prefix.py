import os
import torch
from safetensors.torch import load_file, save_file
import json

src_dir = "medgemma-1.5-4b-pytorch"
dst_dir = "medgemma-1.5-4b-stripped"
os.makedirs(dst_dir, exist_ok=True)

# Copy non-safetensors files
for f in os.listdir(src_dir):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(src_dir, f)):
        import shutil
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

# Process safetensors
st_files = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
for st in st_files:
    print(f"Processing {st}...")
    weights = load_file(os.path.join(src_dir, st), device="cpu")
    new_weights = {}
    for k, v in weights.items():
        if k.startswith("language_model.model."):
            new_key = k.replace("language_model.model.", "")
            new_weights[new_key] = v
        elif k.startswith("language_model.lm_head"):
             # For some models lm_head might be outside "model."
             new_key = k.replace("language_model.", "")
             new_weights[new_key] = v
        else:
            new_weights[k] = v
    save_file(new_weights, os.path.join(dst_dir, st))

# Update model.safetensors.index.json if it exists
index_path = os.path.join(src_dir, "model.safetensors.index.json")
if os.path.exists(index_path):
    with open(index_path, "r") as f:
        index = json.load(f)
    new_weight_map = {}
    for k, v in index["weight_map"].items():
        if k.startswith("language_model.model."):
            new_key = k.replace("language_model.model.", "")
            new_weight_map[new_key] = v
        elif k.startswith("language_model.lm_head"):
            new_key = k.replace("language_model.", "")
            new_weight_map[new_key] = v
        else:
            new_weight_map[k] = v
    index["weight_map"] = new_weight_map
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

print("Prefix stripping complete.")
