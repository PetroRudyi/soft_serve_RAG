import os
import json
import requests
from urllib.parse import urlparse
import re
from config import Config
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# Load articles
with open(f'../{Config.parsed_full_data_path}', 'r', encoding='utf-8') as f:
    articles = json.load(f)

# Prepare output directory
output_dir = f'../{Config.image_dataset_path}'
os.makedirs(output_dir, exist_ok=True)

def slugify(text):
    if text is not None:
        text = text.lower().strip().replace(' ', '_')
    else:
        text = 'no_text'
    return re.sub(r'[^a-zA-Z0-9_\-]', '', text)

def filtration_by_filename(photo_filename: str) -> bool:
    for rule in Config.filtration_keywords:
        if all(keyword in photo_filename for keyword in rule):
            return True
    return False

# Download images
uris = []
metadata = []

for article in tqdm(articles):
    url = article.get('url', 'None')
    issue = article.get('issue', 'unknown')
    title_slug = slugify(article.get('title', 'no_title'))

    for idx, img_url in enumerate(article.get('images', [])):
        try:
            parsed_url = urlparse(img_url)
            img_name = os.path.basename(parsed_url.path)
            img_ext = os.path.splitext(img_name)[-1] or '.jpg'
            img_file_slug = slugify(os.path.splitext(img_name)[0])
            filename = f'issue_{issue}_title_{title_slug}_img_{idx}_{img_file_slug}{img_ext}'

            if filtration_by_filename(filename):
                #print('filtrated:', filename)
                continue

            filepath = os.path.join(output_dir, filename)

            # Step 1: Download
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()

            # Step 2: Check Content-Type is image
            if 'image' not in response.headers.get('Content-Type', ''):
                continue

            # Step 3: Validate the image can be opened
            try:
                img = Image.open(BytesIO(response.content))
                img.verify()  # Checks for corrupted image
            except Exception:
                continue  # Skip invalid images

            # Step 4: Save the validated image
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Save metadata
            uris.append(filename)
            metadata.append({'url': url, 'issue': issue, 'title': title_slug})

        except Exception:
            # Skip image if any error occurs
            continue

# Save metadata to JSON
export_data = {
    "uris": uris,
    "metadata": metadata
}

with open(f'../{Config.dataset_path}/photo_data.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, ensure_ascii=False, indent=4)
