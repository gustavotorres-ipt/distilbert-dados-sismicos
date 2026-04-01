from transformers import AutoTokenizer
import config
import os


LOCAL_TOKENIZER_DIR = f"{config.USED_MODEL}_tokenizer"


def load_tokenizer_bert():
    # ----------------------------
    # Step 1: Define local cache folder
    # ----------------------------
    # Make sure folder exists
    os.makedirs(LOCAL_TOKENIZER_DIR, exist_ok=True)

    # ----------------------------
    # Step 2: Download and save the tokenizer (do this once online)
    # ----------------------------
    if not os.listdir(LOCAL_TOKENIZER_DIR):
        print("Downloading tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(config.USED_MODEL)
        
        # Add special tokens if needed
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Save tokenizer locally
        tokenizer.save_pretrained(LOCAL_TOKENIZER_DIR)
        print(f"Tokenizer saved to {LOCAL_TOKENIZER_DIR}")

    else:
        print(f"Tokenizer already exists at {LOCAL_TOKENIZER_DIR}")
    # ----------------------------
    # Step 3: Load tokenizer from local folder (offline)
    # ----------------------------
    print("Loading tokenizer from local folder...")
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_TOKENIZER_DIR, local_files_only=True)

    # Add pad token if not already in tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
