# MedGemma to LiteRT Agent Rules

- **Objective**: Convert `google/medgemma-1.5-4b-it` to a 4-bit quantized LiteRT (.bin) bundle. The source comes from https://huggingface.co/google/medgemma-1.5-4b-it
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
  6. **CUDA Compatibility**: Ensure all installed Python libraries (especially `torch`, `jax`, and `mediapipe`) are compatible with the system's CUDA version (CUDA 11.8).
  7. **Execution Mode**: Execute the task with `--yolo` mode enabled for accelerated processing.
