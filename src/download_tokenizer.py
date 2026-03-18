import os
import json
import logging
import glob
from huggingface_hub import snapshot_download
from utils import timer_decorator

BASE_DIR = "/"
TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]

def setup_env():
    if os.getenv("TESTING_DOWNLOAD") == "1":
        BASE_DIR = "tmp"
        os.makedirs(BASE_DIR, exist_ok=True)
        os.environ.update({
            "HF_HOME": f"{BASE_DIR}/hf_cache",
            "MODEL_NAME": "openchat/openchat-3.5-0106",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        })

@timer_decorator
def download_tokenizer(name, revision, cache_dir):
    pattern_sets = TOKENIZER_PATTERNS
    try:
        for pattern_set in pattern_sets:
            path = snapshot_download(name, revision=revision, cache_dir=cache_dir, 
                                    allow_patterns=pattern_set)
            for pattern in pattern_set:
                if glob.glob(os.path.join(path, pattern)):
                    logging.info(f"Successfully downloaded tokenizer files matching {pattern}.")
                    return path
    except ValueError:
        raise ValueError(f"No patterns matching {pattern_sets} found for download.")

if __name__ == "__main__":
    setup_env()
    cache_dir = os.getenv("HF_HOME")
    model_name = os.getenv("MODEL_NAME")
    model_revision = os.getenv("MODEL_REVISION") or None
    tokenizer_name = os.getenv("TOKENIZER_NAME") or model_name
    tokenizer_revision = os.getenv("TOKENIZER_REVISION") or model_revision
   
    tokenizer_path = download_tokenizer(tokenizer_name, tokenizer_revision, cache_dir)
    
    metadata = {
        "TOKENIZER_NAME": tokenizer_path,
        "TOKENIZER_REVISION": tokenizer_revision
    }
    
    with open(f"{BASE_DIR}/tokenizer_args.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if v not in (None, "")}, f)
    
    logging.info(f"Tokenizer downloaded to: {tokenizer_path}")