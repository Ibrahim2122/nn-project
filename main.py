# BERT Sentiment Classification Project

# 1. Install required libraries
# pip install transformers datasets scikit-learn torch

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 2. Load dataset
raw_datasets = load_dataset("imdb")

# 3. Load tokenizer and tokenize data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

encoded_datasets = raw_datasets.map(tokenize_function, batched=True)

# 4. Prepare for PyTorch
encoded_datasets = encoded_datasets.rename_column("label", "labels")
encoded_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset = encoded_datasets["train"].shuffle(seed=42).select(range(2000))
test_dataset = encoded_datasets["test"].shuffle(seed=42).select(range(1000))

# 5. Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 9. Train and evaluate
trainer.train()
trainer.evaluate()

