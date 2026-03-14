import os
import psutil
import gc
import sys

# Redirect temporary directory to current workspace to avoid filling /var/tmp
os.environ["TMPDIR"] = os.path.abspath(".")
os.environ["TEMP"] = os.path.abspath(".")
os.environ["TMP"] = os.path.abspath(".")

def check_memory():
    mem = psutil.virtual_memory()
    print(f"[Memory] System RAM Usage: {mem.percent}%")
    if mem.percent > 80:
        print("[WARNING] RAM pressure exceeds 80%. Consider creating a swap-file to prevent OOM.")
        gc.collect()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true", help="Skip standard checks and run accelerated path")
    args, unknown = parser.parse_known_args()

    check_memory()
    
    input_ckpt = "./medgemma-1.5-4b-pytorch"
    tflite_out = "./medgemma-1.5-4b-it.tflite"
    task_out = "./medgemma-1.5-4b-it.task"
    
    # If yolo mode is on, skip standard MediaPipe conversion and go to litert-torch
    if not args.yolo:
        print("Initializing MediaPipe Conversion...")
        try:
            from mediapipe.tasks.python.genai import converter
            from mediapipe.tasks.python.genai import bundler
        except ImportError as e:
            print(f"Failed to import MediaPipe GenAI tools: {e}")
            sys.exit(1)

        # First attempt: Try standard MediaPipe GenAI conversion
        config = converter.ConversionConfig(
            input_ckpt=input_ckpt,
            ckpt_format="safetensors",
            model_type="GEMMA3_4B",
            backend="cpu",
            output_dir=".",

            combine_file_only=False,
            vocab_model_file=f"{input_ckpt}/tokenizer.model",
            output_tflite_file=tflite_out,
            attention_quant_bits=4,
            feedforward_quant_bits=4,
            embedding_quant_bits=4
        )
        
        try:
            print("Running convert_checkpoint() with GEMMA3...")
            converter.convert_checkpoint(config)
            print("MediaPipe standard conversion successful!")
            standard_success = True
        except Exception as e:
            print(f"\n[ERROR] Standard MediaPipe conversion failed: {e}")
            standard_success = False
    else:
        print("[YOLO] Skipping standard MediaPipe conversion.")
        standard_success = False

    if not standard_success:
        print("Falling back to ai-edge-torch generative Gemma 3 text model export...")
        
        import torch
        import litert_torch as ai_edge_torch
        import sys
        import os
        
        sys.path.insert(0, os.path.abspath("./ai-edge-torch"))
        
        try:
            from litert_torch.generative.layers import model_config as cfg
            from litert_torch.generative.utilities import model_builder
            from litert_torch.generative.utilities import loader as loading_utils
            from litert_torch.generative.examples.gemma3 import decoder
            from litert_torch.generative.quantize import quant_recipes
            from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
            
            def get_medgemma_config():
                norm_config = cfg.NormalizationConfig(
                    type=cfg.NormalizationType.RMS_NORM,
                    epsilon=1e-06,
                    with_scale=True,
                    zero_centered=True,
                )
                
                # Attention patterns from HF config: 5 sliding, 1 full
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
                    max_seq_len=16,
                    embedding_dim=2560,
                    block_configs=[get_block_config(i) for i in range(34)],
                    final_norm_config=norm_config,
                    embedding_scale=2560**0.5,
                )

            check_memory()
            print("Step 1: Building MedGemma 4B text model structure...")
            
            # Custom TENSOR_NAMES because HF model has "language_model." prefix
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
            print("Step 2: Loading weights into model...")
            text_model = model_builder.build_decoder_only_model(
                checkpoint_path=input_ckpt,
                config=medgemma_config,
                tensor_names=medgemma_tensor_names,
                model_class=decoder.Decoder,
            )
            print("Step 3: Weights loaded. Initializing generative converter...")
            
            from litert_torch.generative.utilities import converter as gen_converter
            from litert_torch.generative.layers import kv_cache as kv_utils
            
            export_config = gen_converter.ExportConfig(
                mask_as_input=True,
                kvcache_layout=kv_utils.KV_LAYOUT_DEFAULT
            )
            
            import threading
            import time
            def heartbeat():
                while True:
                    time.sleep(60)
                    mem = psutil.virtual_memory()
                    disk = psutil.disk_usage('.')
                    print(f"[Heartbeat] Still converting... RAM: {mem.percent}% Disk Avail: {disk.free // (1024*1024)} MB")
                    sys.stdout.flush()
            
            h_thread = threading.Thread(target=heartbeat, daemon=True)
            h_thread.start()

            import gc
            gc.collect()
            output_tflite = None
            try:
                print("Step 4: Starting convert_to_tflite (this may take several minutes)...")
                output_tflite = gen_converter.convert_to_tflite(
                    pytorch_model=text_model,
                    output_path=".",
                    output_name_prefix="medgemma-1.5-4b-it",
                    prefill_seq_len=[1],
                    kv_cache_max_len=2,
                    quantize='dynamic_int8',
                    config=medgemma_config,
                    export_config=export_config
                )
                print("Step 5: Conversion finished.")
            finally:
                print("Cleaning up temporary directories...")
                import glob
                for d in glob.glob("tmp*"):
                    if os.path.isdir(d):
                        import shutil
                        shutil.rmtree(d, ignore_errors=True)
                
                # Rename output_tflite to tflite_out
                if output_tflite and os.path.exists(output_tflite) and os.path.abspath(output_tflite) != os.path.abspath(tflite_out):
                    import shutil
                    shutil.move(output_tflite, tflite_out)
                    print(f"Moved {output_tflite} to {tflite_out}")
                
                if output_tflite:
                    print(f"Successfully exported to {tflite_out}")
        except Exception as cvt_e:
            print(f"ai-edge-torch convert failed: {cvt_e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("Invoking separate process for LiteRT Bundling to avoid C++ ABI conflicts...")
    import subprocess
    bundling_cmd = [
        sys.executable,
        "bundle_medgemma.py",
        "--tflite_model", tflite_out,
        "--tokenizer_model", f"{input_ckpt}/tokenizer.model",
        "--output_filename", task_out
    ]
    try:
        subprocess.run(bundling_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Bundling failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
