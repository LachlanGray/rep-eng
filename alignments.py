import torch
import text_model
import dataset
import hashlib
import os

def get_hash(x):
    return hashlib.md5(x.encode()).hexdigest()

def compute_nearest_neighbors(feats, topk=1):
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    feats = feats / feats.norm(dim=1, keepdim=True)
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


class TextAlignment:
    """
    Calculate alignments between 
    """
    def __init__(self, models:list[str], sequences:list[str]):

        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        seqs_id = str(get_hash("".join(sequences)))

        self.encodings = {}

        for model in models:
            print(f"Starting {model}")
            cache_pth = os.path.join(cache_dir, f"{model.split('/')[-1]}_{seqs_id}.pth")
            if os.path.exists(cache_pth):
                self.encodings[model] = torch.load(cache_pth)
                continue

            encs = text_model.meanpool_encode(sequences, model)
            self.encodings[model] = encs

            torch.save(encs, cache_pth)
            print("done")


    def similarities(self, kernel_fn=compute_nearest_neighbors):
        sims = {}
        for model, encs in self.encodings.items():
            sims[model] = kernel_fn(encs, topk=15)

        return sims



if __name__ == "__main__":
    models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B"
    ]

    questions, _ = dataset.get_arc()

    alignment = TextAlignment(models, questions)
    sims = alignment.similarities()
    breakpoint()

