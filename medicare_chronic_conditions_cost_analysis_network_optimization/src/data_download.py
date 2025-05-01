import os
import requests

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
            return f"Downloaded successfully to {save_path}"

    else:
        return f"Failed to download from {url}"

