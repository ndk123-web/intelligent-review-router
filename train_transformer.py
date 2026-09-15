import os
import sys
import json
import torch
import pandas as pd
import numpy as np

# Force UTF-8 stdout encoding for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def train_transformer():
    print("=" * 60)
    print("[INFO] Starting DistilBERT Transformer Fine-Tuning Pipeline...")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch Device: {device}")
    if device == "cuda":
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)} (CUDA Available)")
        print(f"VRAM Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("⚠️ Warning: CUDA GPU not detected, falling back to CPU execution.")
        
    data_path = "data/processed/ecommerce_reviews.csv"
    if not os.path.exists(data_path):
        data_path = "../data/processed/ecommerce_reviews.csv"
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} samples.")
    
    label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    id2label = {v: k for k, v in label_map.items()}
    df['label'] = df['sentiment'].map(label_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"Train set size: {len(train_df)} | Test set size: {len(test_df)}")
    
    MODEL_NAME = "distilbert-base-uncased"
    print(f"Loading pretrained tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_func(batch):
        return tokenizer(batch['review'], padding='max_length', truncation=True, max_length=128)
        
    train_ds = Dataset.from_pandas(train_df[['review', 'label']])
    test_ds = Dataset.from_pandas(test_df[['review', 'label']])
    
    print("Tokenizing datasets...")
    train_encoded = train_ds.map(tokenize_func, batched=True)
    test_encoded = test_ds.map(tokenize_func, batched=True)
    
    print(f"Loading pretrained Transformer model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label_map
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1": f1}
        
    save_dir = "app/models/trained_model/sentiment"
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 32 if device == "cuda" else 16
    training_args = TrainingArguments(
        output_dir="./distilbert_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        fp16=(device == "cuda"),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=test_encoded,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print(f"\n[INFO] Fine-tuning DistilBERT model on {device.upper()}...")
    trainer.train()
    
    print("\nEvaluating model performance on test set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Metrics: Accuracy = {eval_results['eval_accuracy']:.4f}, Weighted F1 = {eval_results['eval_f1']:.4f}")
    
    print(f"\nSaving fine-tuned Transformer model & tokenizer to: {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
        
    print("=" * 60)
    print("[SUCCESS] DistilBERT Fine-Tuning Complete! Model is ready for production inference.")
    print("=" * 60)

if __name__ == "__main__":
    train_transformer()
