import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "mps"


def load_model_and_tokenizer(model_name: str,token):
    """
    Loads model and tokenizer.
    """
    device = torch.device(DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.bfloat16,  # Use half precision
            device_map=device,  # Automatically choose the best device
        )
    print(f"Loaded {model_name}")

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def meanpool_encode(text, model_str):
    model, tokenizer = load_model_and_tokenizer(model_str, HF_TOKEN)
    device = torch.device(DEVICE)
    model.to(device)

    input_tokens = tokenizer(text, return_tensors="pt", padding=True)
    input_tokens = {k: v.to(device) for k, v in input_tokens.items()}

    with torch.no_grad():
        outputs = model(**input_tokens, output_hidden_states=True)


    encoding = outputs.hidden_states[-1]
    encoding = encoding * input_tokens['attention_mask'].unsqueeze(-1) # zero out pad tokens

    meanpool = torch.mean(encoding, dim=1)

    return meanpool


if __name__ == "__main__":
    model_str = "meta-llama/Llama-3.2-1B-Instruct"

    device = torch.device(DEVICE)

    input_text = ["How to make a cake", "I don't know how to make a cake"]

    encodings = meanpool_encode(input_text, model_str)

    breakpoint()
