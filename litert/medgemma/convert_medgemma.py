import os
import psutil
import gc
import sys

def check_memory():
    mem = psutil.virtual_memory()
    print(f"[Memory] System RAM Usage: {mem.percent}%")
    if mem.percent > 80:
        print("[WARNING] RAM pressure exceeds 80%. Consider creating a swap-file to prevent OOM.")
        gc.collect()

def main():
    check_memory()
    
    input_ckpt = "./medgemma-1.5-4b-pytorch"
    tflite_out = "./medgemma-1.5-4b.tflite"
    task_out = "./medgemma-1.5-4b.task"
    
    print("Initializing MediaPipe Conversion...")
    try:
        from mediapipe.tasks.python.genai import converter
        from mediapipe.tasks.python.genai import bundler
    except ImportError as e:
        print(f"Failed to import MediaPipe GenAI tools: {e}")
        sys.exit(1)

    # First attempt: Try standard MediaPipe GenAI conversion (if Gemma 3 / Multimodal is supported)
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
    except Exception as e:
        print(f"\n[ERROR] Standard MediaPipe conversion failed for GEMMA3: {e}")
        try:
            print("Trying model_type='GEMMA_3'...")
            config.model_type = "GEMMA_3"
            converter.convert_checkpoint(config)
            print("MediaPipe standard conversion successful!")
        except Exception as e2:
            print(f"\n[ERROR] Standard MediaPipe conversion failed for GEMMA_3: {e2}")
            print("Falling back to ai-edge-torch PyTorch loading with low_cpu_mem_usage=True...")
            
            import torch
            import litert_torch as ai_edge_torch
            from transformers import AutoModelForImageTextToText
            
            check_memory()
            print("Loading PyTorch model...")
            model = AutoModelForImageTextToText.from_pretrained(
                input_ckpt, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )
            check_memory()
            print("Model loaded into RAM.")
            print("Proceeding to manual ai-edge-torch multimodal quantization...")
            
            # Use quant_recipes
            from litert_torch.generative.quantize import quant_recipes
            from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
            quant_config = quant_recipes.full_dynamic_recipe(
                weight_dtype=Dtype.INT4, 
                granularity=Granularity.BLOCKWISE_32
            )
            
            # For multimodal, we may need specific token/input formatting
            # As a fallback, try direct conversion if signature is complex
            try:
                print("Attempting to convert model...")
                edge_model = ai_edge_torch.convert(model, quant_config=quant_config)
                edge_model.export(tflite_out)
                print(f"Exported to {tflite_out}")
            except Exception as cvt_e:
                print(f"ai-edge-torch convert failed: {cvt_e}")
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
