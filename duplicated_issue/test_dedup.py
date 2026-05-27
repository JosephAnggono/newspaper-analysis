import os
import imagehash
from PIL import Image

def get_phash(img_path):
    try:
        img = Image.open(img_path)
        return imagehash.phash(img)
    except Exception as e:
        return None

# Folder with your downloaded samples
folder = "./the_standard"
hashes = {}
duplicates = []

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    phash = get_phash(path)
    
    if phash:
        # Check if this hash already exists
        for existing_hash, existing_file in hashes.items():
            # If distance is 0, they are visually identical
            if phash - existing_hash == 0:
                duplicates.append((filename, existing_file))
                print(f"✅ DUPLICATE FOUND: {filename} is same as {existing_file}")
                break
        hashes[phash] = filename

if not duplicates:
    print("No duplicates found in this sample.")
else:
    print(f"\nTotal duplicates detected: {len(duplicates)}")

