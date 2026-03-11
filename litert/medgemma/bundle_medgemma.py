import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create MediaPipe LiteRT Bundle")
    parser.add_argument("--tflite_model", type=str, required=True, help="Path to the TFLite model")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="Path to the tokenizer model")
    parser.add_argument("--output_filename", type=str, required=True, help="Path for the output .task file")
    
    args = parser.parse_args()
    
    print(f"Initializing MediaPipe Bundler...")
    try:
        from mediapipe.tasks.python.genai import bundler
    except ImportError as e:
        print(f"Failed to import MediaPipe GenAI bundler: {e}")
        sys.exit(1)

    print(f"Creating LiteRT Bundle: {args.output_filename}")
    b_config = bundler.BundleConfig(
        tflite_model=args.tflite_model,
        tokenizer_model=args.tokenizer_model,
        start_token="<bos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        output_filename=args.output_filename,
        prompt_prefix_user="<start_of_turn>user\n",
        prompt_suffix_user="<end_of_turn>\n<start_of_turn>model\n",
        tflite_vision_encoder="medgemma-1.5-4b-vision.tflite"
    )
    bundler.create_bundle(b_config)
    print(f"Bundle successfully created: {args.output_filename}")

if __name__ == "__main__":
    main()
