import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Target model details
REPO = "Qwen/Qwen2.5-Coder-1.5B"
FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json"
]
LFS_FILES = [
    "model.safetensors"
]
LOCAL_DIR = "./model_weights"
NUM_THREADS = 8
MAX_RETRIES = 5
TIMEOUT = 20

def download_small_file(filename):
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    local_path = os.path.join(LOCAL_DIR, filename)
    print(f"Downloading {filename}...")
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, verify=False, timeout=TIMEOUT)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved {filename}")
                return
            else:
                print(f"Attempt {attempt+1}: Failed to download {filename} (HTTP {response.status_code})")
        except Exception as e:
            print(f"Attempt {attempt+1}: Error downloading {filename}: {e}")
        time.sleep(2)
    raise RuntimeError(f"Could not download {filename} after {MAX_RETRIES} attempts.")

def download_chunk(url, start_byte, end_byte, chunk_idx, temp_paths):
    temp_path = f"{LOCAL_DIR}/model.safetensors.part{chunk_idx}"
    temp_paths[chunk_idx] = temp_path
    
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    expected_size = end_byte - start_byte + 1
    
    # Check if this part is already downloaded completely
    if os.path.exists(temp_path) and os.path.getsize(temp_path) == expected_size:
        print(f"  Chunk {chunk_idx} already complete ({expected_size / (1024*1024):.1f} MB). Skipping.")
        return

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Starting Chunk {chunk_idx} (Attempt {attempt+1})...")
            # If the temp file exists but is incomplete, we can delete it and redownload
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            response = requests.get(url, headers=headers, stream=True, verify=False, timeout=TIMEOUT)
            if response.status_code in [200, 206]:
                with open(temp_path, "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                
                actual_size = os.path.getsize(temp_path)
                if actual_size == expected_size:
                    print(f"  Chunk {chunk_idx} finished successfully ({actual_size / (1024*1024):.1f} MB)!")
                    return
                else:
                    print(f"  Chunk {chunk_idx} mismatch size: got {actual_size}, expected {expected_size}. Retrying...")
            else:
                print(f"  Chunk {chunk_idx} attempt {attempt+1} failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"  Chunk {chunk_idx} attempt {attempt+1} exception: {e}")
        time.sleep(3)
        
    raise RuntimeError(f"Chunk {chunk_idx} failed after {MAX_RETRIES} attempts.")

def download_large_file(filename):
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"
    local_path = os.path.join(LOCAL_DIR, filename)
    
    # Send HEAD request to get file size and redirect URL
    response = requests.head(url, allow_redirects=True, verify=False, timeout=TIMEOUT)
    if response.status_code != 200:
        print(f"Failed to get headers for {filename}: HTTP {response.status_code}")
        return
        
    total_size = int(response.headers.get("content-length", 0))
    download_url = response.url
    print(f"Downloading {filename} ({total_size / (1024*1024*1024):.2f} GB) using {NUM_THREADS} threads...")
    
    chunk_size = total_size // NUM_THREADS
    temp_paths = {}
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for idx in range(NUM_THREADS):
            start_byte = idx * chunk_size
            end_byte = total_size - 1 if idx == NUM_THREADS - 1 else (idx + 1) * chunk_size - 1
            futures.append(
                executor.submit(download_chunk, download_url, start_byte, end_byte, idx, temp_paths)
            )
            
        # Check results
        for f in futures:
            f.result()
            
    print("All chunks downloaded successfully. Merging...")
    with open(local_path, "wb") as outfile:
        for idx in range(NUM_THREADS):
            part_path = temp_paths[idx]
            with open(part_path, "rb") as infile:
                outfile.write(infile.read())
            os.remove(part_path)
            
    print(f"Successfully downloaded and merged {filename}!")

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # Download small config files
    for filename in FILES:
        download_small_file(filename)
        
    # Download large model weight files
    for filename in LFS_FILES:
        download_large_file(filename)
        
    print("\nAll files downloaded to local directory 'model_weights/'.")

if __name__ == "__main__":
    main()
