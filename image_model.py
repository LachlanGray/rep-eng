import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModel
import os

from config import DEVICE

# Define your device (MPS for Apple Silicon, or "cuda" for GPU)
HF_TOKEN = os.getenv("HF_TOKEN")


def load_vision_model_and_feature_extractor(model_name: str, token):
    """
    Loads a vision model and feature extractor.
    """
    print(f"Starting to load vision model: {model_name}")
    device = torch.device(DEVICE)

    # Load feature extractor (for image preprocessing)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, token=token)
    print("Feature extractor loaded.")

    # Load vision model
    print("Starting to load vision model...")
    model = AutoModel.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.bfloat16,  # Use half precision if supported
            device_map=device,  # Automatically choose the best device
        )
    print("Vision model loaded successfully.")

    return model, feature_extractor


def encode_images(model_name: str, token: str, image_list: list):

    # Load model and feature extractor
    model, feature_extractor = load_vision_model_and_feature_extractor(model_name, token)

    model.eval()

    embeddings = []

    for image in image_list:

        if isinstance(image, str):
            image = cv2.imread(image)
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = feature_extractor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # Example: taking the mean of hidden states
            embeddings.append(embedding)

    return embeddings


if __name__ == "__main__":
    model_str = "google/vit-base-patch16-224"

    device = torch.device(DEVICE)

    img_path = "dog.jpg"
    enc = encode_images([img_path], model_str)

    breakpoint()
