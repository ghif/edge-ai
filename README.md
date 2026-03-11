# Edge AI Conversions

This repository contains code and implementations for converting AI/ML foundation models into edge-ready formats. The primary focus is on optimizing and bundling models for efficient execution on resource-constrained devices using frameworks like LiteRT (formerly TensorFlow Lite) and ONNX.

## Project Structure

- **litert/**: Contains implementations for converting foundation models to LiteRT (TFLite) format.
  - **medgemma/**: Specific implementation for converting `google/medgemma-1.5-4b-it` to a 4-bit quantized LiteRT (.bin) bundle.
- **onnx/**: (Planned) Implementations for converting models to ONNX format.

## Key Features

- **Quantization**: Support for various quantization recipes, including 4-bit quantization for large language models.
- **MediaPipe Tasks**: Bundling models into MediaPipe-compatible formats for seamless integration into mobile and web applications.
- **Multimodal Support**: Handling complex model architectures, such as multimodal models with vision and text encoders.

## Example: MedGemma 1.5 Conversion

The `litert/medgemma` directory provides a workflow for converting the MedGemma 1.5 model. This process involves:

1.  Downloading the source model from Hugging Face.
2.  Building and quantizing the vision encoder.
3.  Converting the core language model.
4.  Bundling the components into a final LiteRT task bundle.

