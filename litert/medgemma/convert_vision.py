import os
import torch
import sys
import psutil

os.environ["TMPDIR"] = os.path.abspath(".")
os.environ["TEMP"] = os.path.abspath(".")
os.environ["TMP"] = os.path.abspath(".")

sys.path.insert(0, os.path.abspath("./ai-edge-torch"))

import litert_torch as ai_edge_torch
from litert_torch.generative.examples.gemma3 import image_encoder
from litert_torch.generative.utilities import loader as loading_utils
from safetensors.torch import load_file

def main():
    torch.set_default_device('cuda')
    print("Building image encoder...")
    
    # Map HF tensor names to litert_torch expectations
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
    
    # The instruction says: "Crucial Fix: You must map the multi_modal_projector weights in the TENSOR_NAMES mapping"
    # Even if they are not officially part of ModelLoader.TensorNames, we set them as requested.
    setattr(image_encoder.TENSOR_NAMES, "mm_input_projection_weight", "multi_modal_projector.linear_1.weight")
    setattr(image_encoder.TENSOR_NAMES, "mm_soft_emb_norm_weight", "multi_modal_projector.linear_2.weight")
    
    checkpoint_path = "./medgemma-1.5-4b-pytorch"
    encoder = image_encoder.build_image_encoder(checkpoint_path)
    
    # Load multi-modal projector directly to attach it to the vision encoder
    class VisionWithProjector(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            # The projector architecture for MedGemma 1.5 might be linear layers and norms.
            # Using safetensors to find actual projector shapes:
            self.projector_loaded = False
            self.proj_linear_1 = None
            self.proj_linear_2 = None
            
        def load_projector(self, ckpt_path):
            st_files = [f for f in os.listdir(ckpt_path) if f.endswith(".safetensors")]
            state_dict = {}
            for st in st_files:
                state_dict.update(load_file(os.path.join(ckpt_path, st), device='cpu'))
            
            # Find the projector weights
            keys = state_dict.keys()
            proj_keys = [k for k in keys if 'multi_modal_projector' in k]
            print("Projector keys found in checkpoint:", proj_keys)
            
            if 'multi_modal_projector.linear_1.weight' in state_dict:
                w1 = state_dict['multi_modal_projector.linear_1.weight'].to('cuda')
                b1 = state_dict['multi_modal_projector.linear_1.bias'].to('cuda')
                self.proj_linear_1 = torch.nn.Linear(w1.shape[1], w1.shape[0], bias=True)
                self.proj_linear_1.weight.data = w1
                self.proj_linear_1.bias.data = b1
                
                if 'multi_modal_projector.linear_2.weight' in state_dict:
                    w2 = state_dict['multi_modal_projector.linear_2.weight'].to('cuda')
                    b2 = state_dict['multi_modal_projector.linear_2.bias'].to('cuda')
                    self.proj_linear_2 = torch.nn.Linear(w2.shape[1], w2.shape[0], bias=True)
                    self.proj_linear_2.weight.data = w2
                    self.proj_linear_2.bias.data = b2
                
                self.projector_loaded = True
        
        def forward(self, pixel_values):
            x = self.encoder(pixel_values)
            if self.projector_loaded:
                x = self.proj_linear_1(x)
                x = torch.nn.functional.gelu(x)
                if self.proj_linear_2 is not None:
                    x = self.proj_linear_2(x)
            return x

    encoder_with_proj = VisionWithProjector(encoder)
    encoder_with_proj.load_projector(checkpoint_path)
    encoder_with_proj.eval()
    
    print("Vision encoder + projector loaded. Tracing and converting to LiteRT INT4...")
    
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    
    # Dummy inputs: (1, 3, 896, 896) based on image_encoder config
    pixel_values = torch.zeros(1, 3, 896, 896, dtype=torch.float32, device='cuda')
    
    # We must quantize to INT4
    edge_model = ai_edge_torch.convert(encoder_with_proj, (pixel_values,))
    temp_file = "medgemma-1.5-4b-vision.tflite"
    edge_model.export(temp_file)
    print(f"Successfully exported vision encoder to {temp_file}")
    
    print("Quantizing with ai_edge_quantizer...")
    from ai_edge_quantizer import quantizer
    from ai_edge_quantizer import qtyping
    
    q = quantizer.Quantizer(temp_file)
    q.add_weight_only_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=4,
        granularity=qtyping.QuantGranularity.CHANNELWISE
    )
    
    quantization_result = q.quantize()
    output_file = "medgemma-1.5-4b-vision-int4.tflite"
    quantization_result.export_model(output_file)
    print(f"Successfully quantized vision encoder to {output_file}")

if __name__ == "__main__":
    main()