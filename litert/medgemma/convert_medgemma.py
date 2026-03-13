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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", action="store_true", help="Skip standard checks and run accelerated path")
    args, unknown = parser.parse_known_args()

    check_memory()
    
    input_ckpt = "./medgemma-1.5-4b-pytorch"
    tflite_out = "./medgemma-1.5-4b.tflite"
    task_out = "./medgemma-1.5-4b.task"
    
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
        print("Falling back to ai-edge-torch PyTorch loading with low_cpu_mem_usage=True...")
        
        import torch
        import litert_torch as ai_edge_torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import numpy as np
        from PIL import Image
        
        check_memory()
        print("Loading PyTorch model and processor...")
        model = AutoModelForImageTextToText.from_pretrained(
            input_ckpt, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16 
        ).eval()
        processor = AutoProcessor.from_pretrained(input_ckpt)
        check_memory()
        print("Model and processor loaded into RAM.")
        
        # Create sample inputs for tracing
        print("Creating sample inputs for tracing...")
        dummy_image = Image.fromarray(np.zeros((896, 896, 3), dtype=np.uint8))
        # Use chat template for correct formatting
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        sample_inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
        # Ensure use_cache=False to avoid DynamicCache in output, which causes export errors
        # Note: We don't pass 'use_cache' in sample_args if the converter expects only tensors
        
        # Extract sample args - ONLY TENSORS
        sample_args = (sample_inputs['input_ids'], sample_inputs['pixel_values'])
        
        # Wrapper to avoid DynamicCache in output
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input_ids, pixel_values):
                # Explicitly return only logits
                outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, use_cache=False)
                return outputs.logits

        wrapped_model = ModelWrapper(model).eval()
        
        print("Proceeding to manual ai-edge-torch multimodal quantization...")
        
        # Use quant_recipes
        from litert_torch.generative.quantize import quant_recipes
        from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
        quant_config = quant_recipes.full_dynamic_recipe(
            weight_dtype=Dtype.INT4, 
            granularity=Granularity.BLOCKWISE_32
        )
        
        try:
            print("Attempting to convert wrapped model with sample_args...")
            edge_model = ai_edge_torch.convert(wrapped_model, sample_args, quant_config=quant_config)
            edge_model.export(tflite_out)
            print(f"Exported to {tflite_out}")
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
