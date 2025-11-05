import config
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, AutoModel

def check_embeddings():
    sentences = [
        "This is an image of a seismic section with low noise and high frequency.",
        "a model of a reverse fault.",
        "We observe here a sesimic section. It has a tectonic fault.",
        "This depicts a 2D subsurface model with a horizons displacement along fault. It is low frequency and low noise.",
    ]

    model = AutoModel.from_pretrained(config.OUTPUT_FILE)
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_FILE)

    # len(tokenizer) para conseguir o tamanho do vocabulário.
    # decoded_texts = tokenizer.batch_decode([10, 2000, 500, 8000, 27000], skip_special_tokens=True)

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128)
    #ninputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state[:, 0, :]

        print("Embeddings size:", embeddings.shape[1])

if __name__ == "__main__":
    check_embeddings()
