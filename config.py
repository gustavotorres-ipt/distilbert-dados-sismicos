from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="results_mlm_model",
    eval_strategy="epoch",
    save_total_limit=1,
    greater_is_better=True,
    learning_rate=2e-5,
    num_train_epochs=30,
    per_device_train_batch_size=32,
    weight_decay=0.01,
    save_safetensors=True,
)

USED_MODEL = 'distilbert-base-uncased'
TRAINING_CAPTIONS_FILE = "training_captions.csv"
TEST_CAPTIONS_FILE = "test_captions.csv"
# OUTPUT_FILE ="seismic_distilbert.pt"

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
