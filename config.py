from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="results_mlm_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=30,
    weight_decay=0.01,
    per_gpu_train_batch_size=16,
)

USED_MODEL = 'distilbert-base-uncased'
CAPTIONS_FILE = "captions.csv"
OUTPUT_FILE ="seismic_distilbert.pt"
