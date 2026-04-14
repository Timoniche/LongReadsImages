from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


# Now we load and encode the images
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

# We load 3 images. You can either pass URLs or
# a path on your disc
img_paths = [
    # Dog image
    "/Users/ddulaev/Documents/LongReadImages/data/images/small_dzen_ru/block_0.png",
]

images = [load_image(img) for img in img_paths]

# Map images to the vector space
img_embeddings = img_model.encode(images)

# Now we encode our text:
texts = [
    "A cozy and elegant Georgian cuisine restaurant named Magnolia in Kungaralovo, known to many in Moscow but rarely seen online or advertised. The establishment was founded by Georgy Pipia after moving from Batumi during the 90s using AI technology. Despite its popularity among creative circles and athletes, it remains relatively unknown outside of local knowledge.",
]

text_embeddings = text_model.encode(texts)

# Compute cosine similarities:
cos_sim = util.cos_sim(text_embeddings, img_embeddings)

for text, scores in zip(texts, cos_sim):
    max_img_idx = torch.argmax(scores)
    print("Text:", text)
    print("Score:", scores[max_img_idx] )
    print("Path:", img_paths[max_img_idx], "\n")
