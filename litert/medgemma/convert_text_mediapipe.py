import os
import sys
from mediapipe.tasks.python.genai import converter

input_ckpt = "medgemma-1.5-4b-stripped"
output_tflite = "./medgemma-1.5-4b-text.tflite"

config = converter.ConversionConfig(
    input_ckpt=input_ckpt,
    ckpt_format="safetensors",
    model_type="GEMMA3_4B",
    backend="cpu",
    output_dir=".",
    is_quantized=False,
    vocab_model_file=f"./{input_ckpt}/tokenizer.model",
    output_tflite_file=output_tflite
)

print(f"Starting MediaPipe conversion for {input_ckpt}...")
try:
    converter.convert_checkpoint(config)
    print(f"Successfully converted to {output_tflite}")
except Exception as e:
    print(f"Conversion failed: {e}")
    sys.exit(1)
