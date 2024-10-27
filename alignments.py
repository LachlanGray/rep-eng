import torch
import text_model
import dataset
import hashlib
import os

def get_hash(x):
    return hashlib.md5(x.encode()).hexdigest()

def cosine_nearest_neighbors(feats, topk=1):
    assert feats.ndim == 2, f"Expected feats to be 2D, got {feats.ndim}"
    feats = feats / feats.norm(dim=1, keepdim=True)
    knn = (
        (feats @ feats.T).fill_diagonal_(-1e8).argsort(dim=1, descending=True)[:, :topk]
    )
    return knn


class TextAlignment:
    """
    Embedding alignments for text models
    """
    def __init__(self, models:list[str], sequences:list[str]):
        """
        Args:
            models: list of hf models
            sequences: list of prompt strings to encode
            topk: number of nearest neighbors to compute
        """
        self.models = models

        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        seqs_id = str(get_hash("".join(sequences)))

        self.encodings = {}
        self.sims = {}

        for model in self.models:
            print(f"Starting {model}")
            cache_pth = os.path.join(cache_dir, f"{model.split('/')[-1]}_{seqs_id}.pth")
            if os.path.exists(cache_pth):
                self.encodings[model] = torch.load(cache_pth)
                continue

            encs = text_model.meanpool_encode(sequences, model)
            self.encodings[model] = encs

            torch.save(encs, cache_pth)
            print("done")


    def compute_similarities(self, kernel_fn="cosine", topk=10):
        assert kernel_fn in ["cosine"]

        if kernel_fn == "cosine":
            kernel_fn = lambda x: cosine_nearest_neighbors(x, topk=topk)

        self.sims = {}
        for model, encs in self.encodings.items():
            self.sims[model] = kernel_fn(encs)

        return self.sims

    def alignment_matrix(self, kernel_fn="cosine", topk=10):
        if self.sims == {}:
            self.compute_similarities(kernel_fn, topk)

        n_models = len(self.models)
        alignments = torch.zeros([n_models, n_models])

        for i in range(n_models):
            model_i = self.models[i]
            for j in range(i+1, n_models):
                model_j = self.models[j]

                sims_i = self.sims[model_i]
                sims_j = self.sims[model_j]

                overlaps = [len(set(x.tolist()) & set(y.tolist())) for x, y in zip(sims_i, sims_j)]

                alignments[i,j] = sum(overlaps) / len(overlaps) / topk

        return alignments + alignments.T


if __name__ == "__main__":
    models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B"
    ]

    questions, _ = dataset.get_arc()

    alignment = TextAlignment(models, questions)
    _ = alignment.compute_similarities()
    m = alignment.alignment_matrix()
    breakpoint()

