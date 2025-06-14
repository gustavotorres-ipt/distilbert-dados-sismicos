# https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
import os
import random
import json
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline
import config

tokenizer = AutoTokenizer.from_pretrained(config.USED_MODEL)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token

def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return captions

def generate_captions_csv():
    caption_files = os.listdir("captions")
    random.shuffle(caption_files)

    training_files = []
    n_training = int((1 - config.TEST_SIZE) * len(caption_files))

    training_files = caption_files[:n_training]
    test_files = caption_files[n_training:]

    read_captions_and_save(training_files, config.TRAINING_CAPTIONS_FILE)
    read_captions_and_save(test_files, config.TEST_CAPTIONS_FILE)


def read_captions_and_save(caption_files, output_file):
    captions = []

    for selected_file in tqdm(caption_files):

        file_path = f"./captions/{selected_file}"
        captions +=  read_captions_json(file_path)

    # Uppercase the first letter.
    captions = [f'{cap[0].upper()}{cap[1:]}' for cap in captions]

    with open(output_file, "w") as f:
        file_content = f"text\n{'\n'.join(captions)}"
        f.write(file_content.strip())

    print(f"{output_file} saved.")


def tokenize_function(content):
    return tokenizer(content["text"], padding=True,
                     truncation=True, max_length=128)

def train(model, dataset, data_collator):
    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer


def training_process():
    dataset = load_dataset("csv", data_files=config.TRAINING_CAPTIONS_FILE, split='train[:]')
    dataset = dataset.train_test_split(test_size=0.2)

    tokenized_dataset = dataset.map( tokenize_function, batched=True, num_proc=4,)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained(config.USED_MODEL)

    # samples = [tokenized_dataset["train"][i] for i in range(2)]
    # for sample in samples:
    #     print(sample)
    #     _ = sample.pop("input_ids")
    # for chunk in data_collator(samples)["input_ids"]:
    #     print(f'{tokenizer.decode(chunk)}')
    # breakpoint()

    trainer = train(model, tokenized_dataset, data_collator)
    print(trainer.evaluate())
    trainer.save_model(config.OUTPUT_FILE)
    print(config.OUTPUT_FILE, "saved.")


def test_process():
    mask_filler = pipeline(
        "fill-mask", model=config.OUTPUT_FILE
    )

    text = "This is a [MASK]"
    preds = mask_filler(text)

    for pred in preds:
        print(pred)


def main():
    # Check if csv captions file already exists
    if config.TRAINING_CAPTIONS_FILE not in os.listdir(".") or \
            config.TEST_CAPTIONS_FILE not in os.listdir("."):
        print("Generating captions...")
        generate_captions_csv()

    if config.OUTPUT_FILE not in os.listdir("."):
        print("Perform fine-tuning...")
        training_process()

    print("Testing model...")
    # evaluate(trainer)
    # model = AutoModelForMaskedLM.from_pretrained(config.OUTPUT_FILE)
    #ninputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # test_process()


if __name__ == "__main__":
    main()
