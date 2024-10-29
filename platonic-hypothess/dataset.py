import os
import requests

from bs4 import BeautifulSoup
from datasets import load_dataset

from config import MAX_SAMPLES, DATA_DIR


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


def get_pascal(limit=MAX_SAMPLES):
    base_url = "https://vision.cs.uiuc.edu/pascal-sentences/"
    image_directory = os.path.join(DATA_DIR, "pascal")

    # load data from disk if it exists
    def load_existing():
        jpg_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])
        jpg_files = [os.path.join(image_directory, f) for f in jpg_files]
        text_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.txt')])


        # Remove the last image if the number of images and text files don't match
        name_of = lambda x: x.split('.')[0]
        if name_of(text_files[-1]) != name_of(jpg_files[-1]):
            jpg_files.pop()

        text = []
        for f in text_files:
            with open(os.path.join(image_directory, f), 'r') as file:
                text.append(file.read())

        return jpg_files, text

    # return images and text if theres enough
    images, text = [], []
    if os.path.exists(image_directory):
        images, text = load_existing()
        if len(images) >= limit:
            return images[:limit], text[:limit]

    os.makedirs(image_directory, exist_ok=True)

    # Send a request to the website
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Loop through all <tr> elements to find images and descriptions
    for tr in soup.find_all('tr'):
        # Find the image tag
        img_tag = tr.find('img')
        if img_tag:
            # Get the image source URL
            img_src = img_tag['src']
            img_url = base_url + img_src

            # Find the nested table with descriptions
            descriptions = []
            nested_table = tr.find_all('td')[1].find('table') if len(tr.find_all('td')) > 1 else None
            if nested_table:
                # Get all <td> elements within the nested <table> and extract the descriptions
                for desc_td in nested_table.find_all('td'):
                    description = desc_td.get_text(strip=True)
                    if description:
                        descriptions.append(description)

            # Download and save the image
            img_name = os.path.join(image_directory, img_src.split('/')[-1])
            img_data = requests.get(img_url).content

            with open(img_name, 'wb') as f:
                f.write(img_data)

            # Print the image name and its associated descriptions
            print(f"saved {img_name}")
            desc_name = img_name.replace('.jpg', '.txt')
            with open(desc_name, 'w') as f:
                for desc in descriptions:
                    f.write(desc + '\n')

    return load_existing()

if __name__ == "__main__":
    images, text = get_pascal()
    breakpoint()
