import os
import torch
import litert_torch as ai_edge_torch

# We add the downloaded repo to sys.path so we can import the gemma3 tools
import sys
sys.path.insert(0, os.path.abspath("./ai-edge-torch"))

from litert_torch.generative.examples.gemma3 import image_encoder
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

def main():
    print("Building image encoder...")
    # Map HF tensor names to litert_torch expectations
    from litert_torch.generative.utilities import loader as loading_utils
    image_encoder.TENSOR_NAMES = loading_utils.ModelLoader.TensorNames(
        ff_up_proj="vision_tower.vision_model.encoder.layers.{}.mlp.fc1",
        ff_down_proj="vision_tower.vision_model.encoder.layers.{}.mlp.fc2",
        attn_query_proj="vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj",
        attn_key_proj="vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj",
        attn_value_proj="vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj",
        attn_output_proj="vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj",
        pre_attn_norm="vision_tower.vision_model.encoder.layers.{}.layer_norm1",
        pre_ff_norm="vision_tower.vision_model.encoder.layers.{}.layer_norm2",
        embedding="vision_tower.vision_model.embeddings.patch_embedding",
        embedding_position="vision_tower.vision_model.embeddings.position_embedding.weight",
        final_norm="vision_tower.vision_model.post_layernorm",
    )
    
    # Load the vision encoder using the builder from ai-edge-torch
    checkpoint_path = "./medgemma-1.5-4b-pytorch"
    encoder = image_encoder.build_image_encoder(checkpoint_path)
    
    print("Vision encoder loaded. Tracing and converting to LiteRT...")
    # Create dummy inputs: (batch_size * num_media, c, h, w)
    # The get_image_encoder_config says image_size=896
    pixel_values = torch.zeros(1, 3, 896, 896, dtype=torch.float32)
    
    # Bypass quantization here; we will quantize later using quantize_vision.py
    # quant_config = quant_recipes.full_dynamic_recipe(
    #     weight_dtype=Dtype.INT4, 
    #     granularity=Granularity.CHANNELWISE
    # )

    try:
        edge_model = ai_edge_torch.convert(encoder, (pixel_values,))
        
        output_file = "medgemma-1.5-4b-vision.tflite"
        edge_model.export(output_file)
        print(f"Successfully exported vision encoder to {output_file}")
    except Exception as e:
        print(f"Error converting vision encoder: {e}")

if __name__ == "__main__":
    main()