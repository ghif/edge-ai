
import numpy as np
import litert_torch as ai_edge_torch
import mediapipe as mp
from mediapipe.tasks.python.genai import bundler
import torch
import cv2

# ==========================================
# MedGemma 1.5 (4B) Multimodal Inference Pipeline
# ==========================================

class MedGemmaMultimodal:
    def __init__(self, vision_model_path, text_model_path, tokenizer_path):
        print("Initializing Vision Encoder...")
        # Load the INT4 quantized SigLIP + Projector LiteRT model
        self.vision_interpreter = ai_edge_torch.Interpreter(vision_model_path)
        self.vision_interpreter.allocate_tensors()
        
        self.vision_input_details = self.vision_interpreter.get_input_details()
        self.vision_output_details = self.vision_interpreter.get_output_details()
        
        print("Initializing Text Decoder...")
        # Normally mediapipe tasks genai API is used to load the bundled .task file
        # If running separately:
        self.text_session = None # Placeholder for MediaPipe GenAI session
        self.tokenizer_path = tokenizer_path
        
    def preprocess_image(self, image_path):
        # MedGemma vision tower expects 896x896 images normalized
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (896, 896))
        img = img.astype(np.float32) / 255.0
        # Normalize with SigLIP mean/std
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = (img - mean) / std
        # HWC to CHW to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
        
    def generate(self, image_path, prompt):
        # 1. Encode Image to Embeddings
        img_tensor = self.preprocess_image(image_path)
        self.vision_interpreter.set_tensor(self.vision_input_details[0]['index'], img_tensor)
        self.vision_interpreter.invoke()
        
        vision_embeddings = self.vision_interpreter.get_tensor(self.vision_output_details[0]['index'])
        print(f"Extracted vision embeddings of shape: {vision_embeddings.shape}")
        
        # 2. Pipeline into Text Prompt
        # In a fully bundled MediaPipe Task, the vision embeddings are concatenated 
        # with the tokenized text embeddings internally. 
        # Here we format the prompt with the <image> placeholder for the wrapper.
        formatted_prompt = f"<start_of_turn>user\n<image>\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        print("Passing multi-modal context to Text Decoder...")
        # response = self.text_session.predict(formatted_prompt, vision_features=vision_embeddings)
        response = "[Mock Output] The patient X-Ray shows normal lung opacity..."
        
        return response

if __name__ == '__main__':
    pipeline = MedGemmaMultimodal(
        'medgemma-1.5-4b-vision-int4.tflite',
        'medgemma-1.5-4b-text-int4.tflite',
        'medgemma-1.5-4b-pytorch/tokenizer.model'
    )
    print(pipeline.generate('sample.jpg', 'What are the findings in this scan?'))
