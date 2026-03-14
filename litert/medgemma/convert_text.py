import os
import psutil
import gc
import sys
import torch
import faulthandler

faulthandler.enable()

os.environ["TMPDIR"] = os.path.abspath(".")
os.environ["TEMP"] = os.path.abspath(".")
os.environ["TMP"] = os.path.abspath(".")

import litert_torch as ai_edge_torch
from litert_torch.generative.layers import model_config as cfg
from litert_torch.generative.utilities import model_builder
from litert_torch.generative.utilities import loader as loading_utils
from litert_torch.generative.examples.gemma3 import decoder
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

def check_memory():
    mem = psutil.virtual_memory()
    print(f"[Memory] System RAM Usage: {mem.percent}%")
    if mem.percent > 80:
        print("[WARNING] RAM pressure exceeds 80%. Consider creating a swap-file to prevent OOM.")
        gc.collect()

def get_medgemma_config():
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=1e-06,
        with_scale=True,
        zero_centered=True,
    )
    
    attn_patterns = [cfg.AttentionType.LOCAL_SLIDING] * 5 + [cfg.AttentionType.GLOBAL]
    
    def get_block_config(idx: int):
        attn_type = attn_patterns[idx % 6]
        return cfg.TransformerBlockConfig(
            attn_config=cfg.AttentionConfig(
                num_heads=8,
                num_query_groups=4,
                head_dim=256,
                rotary_base=1000000 if attn_type == cfg.AttentionType.GLOBAL else 10000,
                rotary_percentage=1.0,
                qkv_use_bias=False,
                qkv_transpose_before_split=True,
                attn_type=attn_type,
                sliding_window_size=1024 if attn_type == cfg.AttentionType.LOCAL_SLIDING else None,
                query_norm_config=norm_config,
                key_norm_config=norm_config,
            ),
            ff_config=cfg.FeedForwardConfig(
                type=cfg.FeedForwardType.GATED,
                activation=cfg.ActivationConfig(cfg.ActivationType.GELU_TANH),
                intermediate_size=10240,
                pre_ff_norm_config=norm_config,
                post_ff_norm_config=norm_config,
            ),
            pre_attention_norm_config=norm_config,
            post_attention_norm_config=norm_config,
        )

    return cfg.ModelConfig(
        vocab_size=262208,
        num_layers=34,
        max_seq_len=1,
        embedding_dim=2560,
        block_configs=[get_block_config(i) for i in range(34)],
        final_norm_config=norm_config,
        embedding_scale=2560**0.5,
    )

def run_conversion(device='cpu'):
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    # Remove CUDA-specific SDP settings
    check_memory()
    print(f"Step 1: Building MedGemma 4B text model structure on {device}...")
    sys.stdout.flush()
    
    medgemma_tensor_names = loading_utils.ModelLoader.TensorNames(
        ff_up_proj="language_model.model.layers.{}.mlp.up_proj",
        ff_down_proj="language_model.model.layers.{}.mlp.down_proj",
        ff_gate_proj="language_model.model.layers.{}.mlp.gate_proj",
        attn_query_proj="language_model.model.layers.{}.self_attn.q_proj",
        attn_key_proj="language_model.model.layers.{}.self_attn.k_proj",
        attn_value_proj="language_model.model.layers.{}.self_attn.v_proj",
        attn_output_proj="language_model.model.layers.{}.self_attn.o_proj",
        attn_query_norm="language_model.model.layers.{}.self_attn.q_norm",
        attn_key_norm="language_model.model.layers.{}.self_attn.k_norm",
        pre_attn_norm="language_model.model.layers.{}.input_layernorm",
        post_attn_norm="language_model.model.layers.{}.post_attention_layernorm",
        pre_ff_norm="language_model.model.layers.{}.pre_feedforward_layernorm",
        post_ff_norm="language_model.model.layers.{}.post_feedforward_layernorm",
        embedding="language_model.model.embed_tokens",
        final_norm="language_model.model.norm",
        lm_head=None,
    )
    
    medgemma_config = get_medgemma_config()
    input_ckpt = "./medgemma-1.5-4b-pytorch"
    print(f"Loading weights from {input_ckpt}...")
    text_model = model_builder.build_decoder_only_model(
        checkpoint_path=input_ckpt,
        config=medgemma_config,
        tensor_names=medgemma_tensor_names,
        model_class=decoder.Decoder,
    )
    print("Model built successfully.")
    sys.stdout.flush()
    
    print("Step 2: Initialize generative converter...")
    sys.stdout.flush()
    from litert_torch.generative.utilities import converter as gen_converter
    from litert_torch.generative.layers import kv_cache as kv_utils
    
    export_config = gen_converter.ExportConfig(
        mask_as_input=True,
        kvcache_layout=kv_utils.KV_LAYOUT_DEFAULT
    )
    
    gc.collect()
    print("Step 3: Tracing Text Decoder with weight_only_int8 (This will take a while)...")
    sys.stdout.flush()
    output_tflite = gen_converter.convert_to_tflite(
        pytorch_model=text_model,
        output_path=".",
        output_name_prefix="medgemma-1.5-4b-text",
        prefill_seq_len=[1],
        kv_cache_max_len=1,
        quantize='weight_only_int8',
        config=medgemma_config,
        export_config=export_config,
    )
    print("Step 4: Conversion finished.")
    
    import shutil
    if output_tflite and os.path.exists(output_tflite):
        shutil.move(output_tflite, "medgemma-1.5-4b-text-int4.tflite")
        print(f"Successfully exported to medgemma-1.5-4b-text-int4.tflite")

def main():
    try:
        run_conversion('cpu')
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
