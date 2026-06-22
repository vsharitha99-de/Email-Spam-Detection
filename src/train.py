"""
train.py
Loads the labeled SMS dataset, cleans it, vectorizes with TF-IDF,
trains Naive Bayes and Logistic Regression, evaluates both,
and saves the best model + vectorizer to disk for the Streamlit app.

Usage:
    python src/train.py --data data/spam.csv
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV with columns ['label', 'text'].
    Also handles the raw UCI/Kaggle format, which uses columns
    'v1' (label) and 'v2' (text) with extra empty columns.
    """
    df = pd.read_csv(path, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
        df = df[["label", "text"]]

    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    return df.reset_index(drop=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label="spam"),
        "recall": recall_score(y_test, preds, pos_label="spam"),
        "f1": f1_score(y_test, preds, pos_label="spam"),
    }
    return metrics


def print_metrics(metrics):
    print(f"\n{metrics['model']}")
    print("-" * len(metrics["model"]))
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")


def main(data_path: str, test_size: float = 0.2, random_state: int = 42):
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows. Class balance:")
    print(df["label"].value_counts(), "\n")

    print("Cleaning text ...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    print("Vectorizing with TF-IDF ...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    results = []

    print("Training Naive Bayes ...")
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_metrics = evaluate("Naive Bayes", nb, X_test_vec, y_test)
    results.append(nb_metrics)
    print_metrics(nb_metrics)

    print("\nTraining Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_vec, y_train)
    lr_metrics = evaluate("Logistic Regression", lr, X_test_vec, y_test)
    results.append(lr_metrics)
    print_metrics(lr_metrics)

    results_df = pd.DataFrame(results).set_index("model")
    print("\n=== Comparison Table ===")
    print(results_df.round(4))

    best_name = results_df["f1"].idxmax()
    best_model = nb if best_name == "Naive Bayes" else lr
    print(f"\nBest model by F1 score: {best_name}")

    with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODELS_DIR / "model_name.txt", "w") as f:
        f.write(best_name)

    results_df.round(4).to_csv(MODELS_DIR / "metrics.csv")

    print(f"\nSaved vectorizer + best model ({best_name}) to {MODELS_DIR}/")
    print("You're ready to run the Streamlit app.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/spam.csv",
        help="Path to the labeled CSV dataset",
    )
    args = parser.parse_args()
    main(args.data)
