from PIL import Image
import os

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (OSError, IOError):
        return False

dataset_dir = "/root/autodl-tmp/dataset/skinDisease_split/test"
for root, _, files in os.walk(dataset_dir):
    for file in files:
        path = os.path.join(root, file)
        if not is_valid_image(path):
            print(f"Removing corrupted image: {path}")
            os.remove(path)
