import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModel
import os

# Define your device (MPS for Apple Silicon, or "cuda" for GPU)
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "mps"


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


def meanpool_encode(image_path, model_str):
    """
    Loads an image from a path using cv2, processes it, and returns the encoding.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load model and feature extractor
    model, feature_extractor = load_vision_model_and_feature_extractor(model_str, HF_TOKEN)
    model.to(DEVICE)

    # Preprocess the image using the feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get the encoding from the last hidden state
    encoding = outputs.hidden_states[-1]

    # Mean pool the encoding to get a fixed-size vector
    meanpool = torch.mean(encoding, dim=1)

    return meanpool


if __name__ == "__main__":
    model_str = "google/vit-base-patch16-224"

    device = torch.device(DEVICE)

    img_path = "dog.jpg"
    enc = meanpool_encode(img_path, model_str)

    breakpoint()
