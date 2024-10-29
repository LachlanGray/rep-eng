import cv2
import torch
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
import os

from config import DEVICE, TRUST_REMOTE_CODE

# Define your device (MPS for Apple Silicon, or "cuda" for GPU)
HF_TOKEN = os.getenv("HF_TOKEN")


def load_vision_model_and_feature_extractor(model_name: str):
    """
    Loads a vision model and feature extractor.
    """
    print(f"Starting to load vision model: {model_name}")
    device = torch.device(DEVICE)

    # Load feature extractor (for image preprocessing)
    feature_extractor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    try:
        print(f"trying to load model as AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
        )
    except ValueError:
        print(f"trying to load model as AutoModel")
        model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=TRUST_REMOTE_CODE,
            torch_dtype=torch.bfloat16,
        )

    return model, feature_extractor


def encode_images(model_name: str, image_list: list):

    # Load model and feature extractor
    model, processor = load_vision_model_and_feature_extractor(model_name)

    model.eval()

    embeddings = []

    for image in image_list:

        if isinstance(image, str):
            image = cv2.imread(image)
            image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]  # Example: taking the mean of hidden states
            embeddings.append(embedding)

    return embeddings


if __name__ == "__main__":
    # model_str = "google/vit-base-patch16-224"
    # model_str = "microsoft/Phi-3.5-vision-instruct"
    model_str = "openai/clip-vit-base-patch32"

    device = torch.device(DEVICE)

    img_path = "dog.jpg"
    enc = encode_images(model_str, [img_path])

    breakpoint()
