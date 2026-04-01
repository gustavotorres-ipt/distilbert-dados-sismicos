# https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c
import os
import random
import csv
import sys
import json
import argparse
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline
import config
from tokenizer import load_tokenizer_bert


tokenizer = load_tokenizer_bert()

def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return captions

def generate_captions_csv(folder_captions):
    caption_files = os.listdir(folder_captions)
    random.shuffle(caption_files)

    training_files = []
    n_training = int((1 - config.TEST_SIZE) * len(caption_files))

    training_files = caption_files[:n_training]
    test_files = caption_files[n_training:]

    read_captions_and_save(
        training_files, folder_captions, config.TRAINING_CAPTIONS_FILE)
    read_captions_and_save(
        test_files, folder_captions, config.TEST_CAPTIONS_FILE)


def read_captions_and_save(caption_files, folder_captions, output_file):
    captions = []

    for selected_file in tqdm(caption_files):

        file_path = os.path.join(folder_captions, selected_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)["captions"]
            captions += data

    # Uppercase the first letter.
    captions = [f'{cap[0].upper()}{cap[1:]}' for cap in captions]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])  # header
        for cap in captions:
            writer.writerow([cap.strip()])

    print(f"{output_file} saved.")


def tokenize_function(content):
    return tokenizer(
        content["text"], padding=True, truncation=True, max_length=128)

def train(model, dataset, data_collator):
    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        # tokenizer=tokenizer,
    )

    trainer.train()
    return trainer


def perform_training(output_model_name: str):
    dataset = load_dataset(
        "csv", data_files=config.TRAINING_CAPTIONS_FILE, split='train[:]')
    dataset = dataset.train_test_split(test_size=0.1)

    tokenized_dataset = dataset.map( tokenize_function, batched=True, num_proc=4,)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained(config.USED_MODEL)

    trainer = train(model, tokenized_dataset, data_collator)
    print(trainer.evaluate())
    trainer.save_model(output_model_name)
    print(output_model_name, "saved.")


def test_model(model_path):
    mask_filler = pipeline(
        "fill-mask", model=model_path
    )
    textlist = [
        "this image presents a subsurface representation with a [MASK] depositional body.",
        "this depicts a 2D seismic model. It is an uniform [MASK] pattern.",
        "a 2D seismic model. It is a chaotic [MASK] geometry.",
        "we see here a seismic section. It contains imbricated reflection [MASK].",
    ]

    for text in textlist:
        preds = mask_filler(text)

        for pred in preds:
            print(pred)
        print("---------------------------")


def main(args):

    files_in_dir = os.listdir('.')

    # Check if csv captions file already exists
    # if config.TRAINING_CAPTIONS_FILE not in files_in_dir or \
    #         config.TEST_CAPTIONS_FILE not in files_in_dir:
    # if args.generate_csv:
    print("Generating captions...")
    generate_captions_csv(args.folder_captions)

    if args.trained_model is not None: 
        if args.trained_model not in files_in_dir:
            print(f"Error. Model {args.trained_model} not found.")
            sys.exit(1)

        # test_model(args.trained_model)

    if args.output_model is not None:
        print("Perform fine-tuning...")
        perform_training(args.output_model)
        # test_model(args.output_model)

    # print("Testing model...")
    # evaluate(trainer)
    # model = AutoModelForMaskedLM.from_pretrained(config.OUTPUT_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a language model using MLM.')

    parser.add_argument('-f', '--folder_captions', type=str, required=True,
                        help='Folder where captions are stored.')
    # parser.add_argument('-g', '--generate_csv', action='store_true',
    #                     help='Generate csv again.')
    parser.add_argument('-o', '--output_model', type=str, default=None,
                        help='Name to save the trained model.')
    parser.add_argument('-t', '--trained_model', type=str, default=None,
                        help='Path of model already trained.')
    args = parser.parse_args()

    main(args)
