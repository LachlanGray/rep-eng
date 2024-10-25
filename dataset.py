from datasets import load_dataset

from config import MAX_SAMPLES

def get_arc(version="ARC-Easy", limit=MAX_SAMPLES):
    assert version in ["ARC-Challenge", "ARC-Easy"]

    ds_dict = load_dataset("allenai/ai2_arc", version)
    ds = ds_dict["train"][:MAX_SAMPLES]       # use train split

    questions = ds["question"]

    answer_keys = ds["answerKey"]
    choices = ds["choices"]
    answers = []
    for c, k in zip(choices, answer_keys):
        idx = c["label"].index(k)
        answers.append(c["text"][idx])

    return questions, answers


if __name__ == "__main__":
    pairs = get_arc()
    for q, a in zip(*pairs):
        print(q)
        print(a)
        print("---")
