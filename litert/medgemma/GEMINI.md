# Agent Directives: MedGemma 1.5 (4B) to LiteRT Multimodal Conversion

## 1. Primary Objective
Convert the Hugging Face `google/medgemma-1.5-4b-it` (a Vision-Language Model based on Gemma 3) into 4-bit quantized LiteRT format (`.tflite`) for use with MediaPipe. 

## 2. Architectural Strategy: The Split-Conversion
Because MedGemma 1.5 is a multimodal model, standard single-pass conversions will drop the vision components or crash due to Out-of-Memory (OOM) errors. You **MUST** decouple the conversion into two separate scripts:
* **Model A (Vision):** SigLIP Vision Encoder + Multi-Modal Projector.
* **Model B (Text):** Gemma 3 Text Decoder.

## 3. Environment & Hardware Constraints (CRITICAL)
Before executing any conversion scripts, enforce the following environment rules to prevent known crashes:
* **Disk Space Bottleneck (`RESOURCE_EXHAUSTED`):** TensorFlow's intermediate `SavedModel` export requires 20GB+ of free space. You must dynamically set the `TMPDIR`, `TEMP`, and `TMP` environment variables in Python to a high-capacity drive before importing `litert_torch`.
* **Split-Brain Tracing (`FakeTensor` device mismatch):** Ensure all models, tensors, and dummy inputs are explicitly moved to `cuda` (e.g., `torch.set_default_device('cuda')`) before calling `ai_edge_torch.convert()`.
* **Dependencies:** Ensure PyTorch 2.6.x+ and a CUDA-compatible `jaxlib` are installed to avoid `AttributeError: int1` and JAX fallback warnings.

## 4. Execution Plan

### Phase 1: Vision Encoder + Projector Conversion
* **File:** `convert_vision.py`
* **Task:** Load the `vision_tower` using `ai-edge-torch`'s `image_encoder` builder.
* **Crucial Fix:** You must map the `multi_modal_projector` weights in the `TENSOR_NAMES` mapping so that the vision features are properly projected into the LLM's embedding space. 
* **Mapping Example:** Map `multi_modal_projector.mm_input_projection_weight` and `multi_modal_projector.mm_soft_emb_norm.weight`.
* **Output:** `medgemma-1.5-4b-vision-int4.tflite`

### Phase 2: Text Decoder Conversion
* **File:** `convert_text.py`
* **Task:** Load the `language_model` backbone using `build_decoder_only_model()`.
* **Quantization:** perform the quantization by the following steps: quantize to 8B (weight only) through ai-edge-torch, and then apply INT4 later during MediaPipe bundling phase.
* **Output:** `medgemma-1.5-4b-text-int4.tflite`

### Phase 3: Bundling / Integration
* **File:** `bundle_model.py` (or instructions for Hugging Face Space deployment).
* **Task:** Use the MediaPipe GenAI bundler to package the two `.tflite` files and the `tokenizer.model` into a unified `.task` or `.bin` package, or draft the Python inference wrapper that pipelines the vision output tensor directly into the text prompt.

## 5. Agent Workflow & Self-Correction
* **Plan:** Verify hardware and paths.
* **Act:** Write or modify the conversion scripts sequentially (Vision first, then Text).
* **Observe:** Monitor RAM and Disk usage during the tracing phase.
* **Reflect:** If a `SaveV2` error occurs, verify `TMPDIR` is correctly applied.

# MedGemma to LiteRT Agent Rules

- **Objective**: Convert `google/medgemma-1.5-4b-it` to a 4-bit quantized LiteRT (.task) bundle. The source comes from https://huggingface.co/google/medgemma-1.5-4b-it
- **Tools Priority**: Use `mediapipe` for the final bundle and `ai-edge-torch` if a custom SigLIP conversion is needed.
- **Constraints**:
  - Always check available disk space before downloading weights.
  - Use `low_cpu_mem_usage=True` to prevent OOM errors.
  - Verify every output file using the `interpreter` check.
- **Workflow**: Plan -> Install Deps -> Convert -> Verify -> Document.

# Agent Directives: MedGemma 1.5 LiteRT Conversion

- **Environment**: linux with CUDA GPU
- **Virtual Environment**: Use conda environment "mgenv".
- **Target**: MediaPipe Task Bundle (.task) with 4-bit quantization.
- **Rules**:
  1. Use `hf-hub` to verify the `google/medgemma-1.5-4b-it` repo structure before downloading.
  2. Use `web-search` or context7 to find the 2026 `litert-torch` quantization recipes specifically for Gemma 3.
  3. **Critical**: MedGemma 1.5 has a 3D SigLIP encoder. If the standard converter fails, the agent must check `ai-edge-torch` for the "Multimodal Task" export path.
  4. Monitor memory using `system-monitor`. If RAM pressure exceeds 80%, suggest a swap-file or `low_cpu_mem_usage`.
  5. Final converted models are stored with prefix `medgemma-1.5-4b-it*`.
  6. **CUDA Compatibility**: Ensure all installed Python libraries (especially `torch`, `jax`, and `mediapipe`) are compatible with the system's CUDA version (CUDA 12.4).
  7. **Execution Mode**: Execute the task with `--yolo` mode enabled for accelerated processing.
  8. **Conversion Device**: use cuda mode first for every conversion attempt. if it fails, use cpu mode.
