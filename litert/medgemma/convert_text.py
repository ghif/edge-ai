import os
import gc
import sys
import torch

# Force Temp Dir to current folder (Ensure you have 20GB+ free here)
os.environ["TMPDIR"] = os.path.abspath(".")
os.environ["TEMP"] = os.path.abspath(".")
os.environ["TMP"] = os.path.abspath(".")

import litert_torch as ai_edge_torch
from litert_torch.generative.layers import model_config as cfg
from litert_torch.generative.utilities import model_builder
from litert_torch.generative.utilities import loader as loading_utils
from litert_torch.generative.examples.gemma3 import decoder
from litert_torch.generative.utilities import converter as gen_converter
from litert_torch.generative.layers import kv_cache as kv_utils

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

def main():
    # CRITICAL FIX 1: Must trace in FP32 so the MLIR compiler doesn't crash on the embedding lookup
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    
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
    
    print("Building MedGemma 4B Text Decoder in FP32...")
    text_model = model_builder.build_decoder_only_model(
        checkpoint_path="./medgemma-1.5-4b-pytorch",
        config=get_medgemma_config(),
        tensor_names=medgemma_tensor_names,
        model_class=decoder.Decoder,
    )
    
    export_config = gen_converter.ExportConfig(
        mask_as_input=True,
        kvcache_layout=kv_utils.KV_LAYOUT_DEFAULT
    )
    
    gc.collect()

    print("Tracing and Quantizing to INT4 Block32 (This will take 15+ minutes)...")
    # CRITICAL FIX 2: Use the exact official Enum for Gemma 3 Quantization
    output_tflite = gen_converter.convert_to_tflite(
        pytorch_model=text_model,
        output_path=".",
        output_name_prefix="medgemma-1.5-4b-text",
        prefill_seq_len=[128], # Expanded to match standard Gemma 3 export constraints
        kv_cache_max_len=512,
        quantize=gen_converter.QuantizationName.DYNAMIC_INT4_BLOCK32,
        config=get_medgemma_config(),
        export_config=export_config,
    )
    
    if output_tflite and os.path.exists(output_tflite):
        import shutil
        final_name = "medgemma-1.5-4b-text-int4.tflite"
        shutil.move(output_tflite, final_name)
        print(f"\n[SUCCESS] Model exported to {final_name}")

if __name__ == "__main__":
    main()