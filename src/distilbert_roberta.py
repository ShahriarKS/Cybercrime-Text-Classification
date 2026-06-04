# -*- coding: utf-8 -*-
"""
Fixed DistilBERT + RoBERTa binary cybercrime text classification script.

Main fixes:
1. Correct binary category mapping so "Non Financial Cybercrime" is not mislabeled as Financial Fraud.
2. Uses macro_f1 and weighted_f1 explicitly instead of default binary F1.
3. Uses tokenized validation dataset consistently for RoBERTa.
4. Removes hardcoded metric dependency for final reports; prints classification report from actual predictions.
"""

import gc
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


DATA_PATH = "/content/datasetCrime.csv"
RANDOM_STATE = 42
MAX_LENGTH = 512
NUM_LABELS = 2

LABEL_NAMES = ["Non Financial Cybercrime", "Financial Fraud"]
LABEL_MAP = {
    "Non Financial Cybercrime": 0,
    "Financial Fraud": 1,
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_binary_category(cat):
    """
    Important: check non-financial first.
    Otherwise "Non Financial Cybercrime" contains the word "financial"
    and may be wrongly labeled as Financial Fraud.
    """
    cat = str(cat).lower().strip()

    if "non financial" in cat or "non-financial" in cat or "nonfinancial" in cat:
        return "Non Financial Cybercrime"
    elif "financial" in cat or "fraud" in cat:
        return "Financial Fraud"
    else:
        return "Non Financial Cybercrime"


def prepare_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, on_bad_lines="skip", engine="python")

    df = df.dropna(subset=["crimeaditionalinfo", "category"])
    df = df.drop_duplicates()

    df["clean_text"] = df["crimeaditionalinfo"].apply(clean_text)
    df["binary_category"] = df["category"].apply(make_binary_category)
    df["label"] = df["binary_category"].map(LABEL_MAP)

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 60% train, 20% validation, 20% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val_df["label"],
    )

    return train_df, val_df, test_df


def tokenize_data(train_df, val_df, test_df, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples["clean_text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Keep only fields needed by Trainer
    keep_columns = ["input_ids", "attention_mask", "label"]
    tokenized_train = tokenized_train.remove_columns(
        [c for c in tokenized_train.column_names if c not in keep_columns]
    )
    tokenized_val = tokenized_val.remove_columns(
        [c for c in tokenized_val.column_names if c not in keep_columns]
    )
    tokenized_test = tokenized_test.remove_columns(
        [c for c in tokenized_test.column_names if c not in keep_columns]
    )

    return tokenizer, tokenized_train, tokenized_val, tokenized_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Label order: 0 = Non Financial, 1 = Financial Fraud
        # Adjust these weights only if you intentionally want to favor one class.
        class_weights = torch.tensor([1.5, 1.0], dtype=torch.float).to(model.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def get_training_args(output_dir, epochs):
    try:
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )
    except TypeError:
        # For older transformers versions where eval_strategy is called evaluation_strategy
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )


def train_and_evaluate(model_name, output_dir, epochs, train_df, val_df, test_df):
    print(f"\n========== Training {model_name} ==========")

    tokenizer, tokenized_train, tokenized_val, tokenized_test = tokenize_data(
        train_df, val_df, test_df, model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
    )

    training_args = get_training_args(output_dir=output_dir, epochs=epochs)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = np.array(tokenized_test["label"])

    print(f"\n--- {model_name} Test Performance ---")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    results = {
        "model": model_name,
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "non_financial_precision": classification_report(
            labels, preds, target_names=LABEL_NAMES, output_dict=True
        )["Non Financial Cybercrime"]["precision"],
        "financial_fraud_recall": classification_report(
            labels, preds, target_names=LABEL_NAMES, output_dict=True
        )["Financial Fraud"]["recall"],
    }

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def plot_comparison(results):
    models = [r["model"].replace("-base-uncased", "").replace("roberta-base", "RoBERTa-base") for r in results]
    accuracy_scores = [r["accuracy"] for r in results]
    macro_f1_scores = [r["macro_f1"] for r in results]
    non_fin_precision = [r["non_financial_precision"] for r in results]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, accuracy_scores, width, label="Overall Accuracy")
    rects2 = ax.bar(x, macro_f1_scores, width, label="Macro Avg F1-Score")
    rects3 = ax.bar(x + width, non_fin_precision, width, label="Non-Financial Precision")

    ax.set_ylabel("Scores (0.0 to 1.0)")
    ax.set_title("Transformer Models Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.70, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.show()


def main():
    print("GPU Active:", torch.cuda.is_available())

    train_df, val_df, test_df = prepare_data(DATA_PATH)

    print("\nClass distribution after fixed mapping:")
    print(pd.concat([train_df, val_df, test_df])["binary_category"].value_counts())

    results = []

    results.append(
        train_and_evaluate(
            model_name="distilbert-base-uncased",
            output_dir="./distilbert_results",
            epochs=2,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
    )

    results.append(
        train_and_evaluate(
            model_name="roberta-base",
            output_dir="./roberta_results",
            epochs=3,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
    )

    print("\n========== Final Summary ==========")
    for r in results:
        print(
            f"{r['model']} | "
            f"Accuracy: {r['accuracy']:.4f} | "
            f"Non-Fin Precision: {r['non_financial_precision']:.4f} | "
            f"Financial Fraud Recall: {r['financial_fraud_recall']:.4f} | "
            f"Macro F1: {r['macro_f1']:.4f} | "
            f"Weighted F1: {r['weighted_f1']:.4f}"
        )

    plot_comparison(results)


if __name__ == "__main__":
    main()
