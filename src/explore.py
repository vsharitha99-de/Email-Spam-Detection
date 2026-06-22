"""
explore.py
Quick exploratory data analysis on the SMS dataset before modeling.
Run this first to sanity check the data.

Usage:
    python src/explore.py --data data/spam.csv
"""

import argparse

import pandas as pd

from train import load_data
from preprocess import clean_text


def main(data_path: str):
    df = load_data(data_path)

    print(f"Total messages: {len(df)}")
    print("\nClass balance:")
    print(df["label"].value_counts())
    print("\nClass balance (%):")
    print((df["label"].value_counts(normalize=True) * 100).round(1))

    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    print("\nMessage length by class (characters):")
    print(df.groupby("label")["char_count"].describe()[["mean", "min", "max"]].round(1))

    print("\nMessage length by class (words):")
    print(df.groupby("label")["word_count"].describe()[["mean", "min", "max"]].round(1))

    df["clean_text"] = df["text"].apply(clean_text)

    print("\nMost common words in spam:")
    spam_words = " ".join(df[df["label"] == "spam"]["clean_text"]).split()
    print(pd.Series(spam_words).value_counts().head(10))

    print("\nMost common words in ham:")
    ham_words = " ".join(df[df["label"] == "ham"]["clean_text"]).split()
    print(pd.Series(ham_words).value_counts().head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/spam.csv")
    args = parser.parse_args()
    main(args.data)
