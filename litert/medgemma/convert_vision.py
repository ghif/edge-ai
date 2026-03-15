import os
import torch
import sys
import psutil

os.environ["TMPDIR"] = os.path.abspath(".")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TEMP"] = os.path.abspath(".")
os.environ["TMP"] = os.path.abspath(".")

import litert_torch as ai_edge_torch
from litert_torch.generative.examples.gemma3 import image_encoder
from litert_torch.generative.utilities import loader as loading_utils
from litert_torch.generative.quantize import quant_recipes
from safetensors.torch import load_file

def main():
    # Force CPU tracing
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    # CUDA is disabled for this conversion
    
    print("Building image encoder on CPU...")
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
    
    checkpoint_path = "./medgemma-1.5-4b-pytorch"
    encoder = image_encoder.build_image_encoder(checkpoint_path)
    
    class VisionWithProjector(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.projector_loaded = False
            self.mm_soft_emb_norm_weight = None
            self.mm_input_projection_weight = None
            
        def load_projector(self, ckpt_path):
            st_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
            state_dict = {}
            for st in st_files:
                state_dict.update(load_file(os.path.join(ckpt_path, st), device='cpu'))
            
            k_norm = 'multi_modal_projector.mm_soft_emb_norm.weight'
            k_proj = 'multi_modal_projector.mm_input_projection_weight'
            
            if k_norm in state_dict and k_proj in state_dict:
                self.mm_soft_emb_norm_weight = torch.nn.Parameter(state_dict[k_norm].clone().detach().to('cpu'))
                self.mm_input_projection_weight = torch.nn.Parameter(state_dict[k_proj].clone().detach().to('cpu'))
                self.projector_loaded = True
                print("Projector weights loaded successfully.")
            else:
                print("Projector weights not found!")
                
        def forward(self, pixel_values):
            x = self.encoder(pixel_values)
            if self.projector_loaded:
                x = torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=self.mm_soft_emb_norm_weight, bias=None)
                x = torch.matmul(x, self.mm_input_projection_weight)
            return x

    encoder_with_proj = VisionWithProjector(encoder)
    encoder_with_proj.load_projector(checkpoint_path)
    encoder_with_proj.to(torch.float32)
    encoder_with_proj.to('cpu')
    encoder_with_proj.eval()
    encoder_with_proj.requires_grad_(False)
    
    print("Tracing vision encoder + projector...")
    pixel_values = torch.zeros(1, 3, 896, 896, dtype=torch.float32, device='cpu')
    try:
        edge_model = ai_edge_torch.convert(encoder_with_proj, (pixel_values,))
        output_file = "medgemma-1.5-4b-vision.tflite"
        edge_model.export(output_file)
        print(f"Successfully exported unquantized vision model to {output_file}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()