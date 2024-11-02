import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_model_and_tokenizer(model_name: str):
    """
    Loads model and tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto", # TODO: generation
        )
    print(f"Loaded {model_name}")

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_submodule(model, key):
    """
    Traverse the model to get the submodule by its state dict key
    """
    keys = key.split(".")
    sub_module = model

    # Traverse the hierarchy to reach the target module
    for k in keys:  # Go through all parts of the key
        if hasattr(sub_module, k):
            # Regular attribute access for typical modules
            sub_module = getattr(sub_module, k)
        elif isinstance(sub_module, torch.nn.ModuleList) or isinstance(sub_module, torch.nn.ModuleDict):
            # If the sub_module is a ModuleList, try accessing by index
            try:
                k_index = int(k)  # Convert to integer for ModuleList
                sub_module = sub_module[k_index]
            except ValueError:
                # For ModuleDict, use the string key directly
                sub_module = sub_module[k]
        else:
            raise KeyError(f"Key part '{k}' not found in model.")

    return sub_module

def replace_submodule(model, key, new_module):
    keys = key.split(".")
    sub_module = model
    for k in keys[:-1]:  # Traverse until the second-to-last key
        if hasattr(sub_module, k):
            sub_module = getattr(sub_module, k)
        elif isinstance(sub_module, nn.ModuleList) or isinstance(sub_module, nn.ModuleDict):
            try:
                k_index = int(k)
                sub_module = sub_module[k_index]
            except ValueError:
                sub_module = sub_module[k]
        else:
            raise KeyError(f"Key part '{k}' not found in model.")

    # Replace the submodule
    last_key = keys[-1]
    if isinstance(sub_module, nn.ModuleList) or isinstance(sub_module, nn.ModuleDict):
        sub_module[int(last_key) if last_key.isdigit() else last_key] = new_module
    else:
        setattr(sub_module, last_key, new_module)

