import sys
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping

def main():
    input_tflite = "medgemma-1.5-4b-vision.tflite"
    output_tflite = "medgemma-1.5-4b-vision-int4.tflite"
    
    print(f"Loading {input_tflite}...")
    
    try:
        q = quantizer.Quantizer(input_tflite)
        q.add_weight_only_config(
            regex=".*",
            operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
            num_bits=4,
            granularity=qtyping.QuantGranularity.CHANNELWISE
        )
        
        print("Applying INT4 quantization...")
        quantization_result = q.quantize()
        
        quantization_result.export_model(output_tflite)
        
        print(f"Successfully quantized vision encoder to {output_tflite}")
    except Exception as e:
        print(f"Quantization failed: {e}")

if __name__ == "__main__":
    main()
