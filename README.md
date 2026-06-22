# Spam Email/SMS Classifier

A text classifier that learns to tell spam from legitimate messages (ham), built with
scikit-learn and served through a Streamlit web app.

## What's in here

```
spam-classifier/
├── data/
│   └── sample_spam.csv      # placeholder dataset (see "Get the real dataset" below)
├── src/
│   ├── preprocess.py        # text cleaning (lowercase, strip punctuation/numbers, stop words)
│   ├── explore.py           # quick EDA: class balance, message length, top words
│   └── train.py             # trains Naive Bayes + Logistic Regression, evaluates, saves best model
├── app/
│   └── app.py                # Streamlit app for live predictions
├── models/                   # created after training: saved vectorizer + model + metrics
└── README.md
```

## ⚠️ Get the real dataset first

This repo ships with `data/sample_spam.csv`, a small 115-message placeholder set
I wrote by hand so the code runs out of the box. **Swap it for the real dataset**
before you submit anything:

1. Go to the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
   or [Kaggle's mirror](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
2. Download it, you'll get a file usually named `spam.csv` with columns `v1` (label) and `v2` (text)
3. Drop it into `data/spam.csv`
4. `train.py` already handles both the `v1/v2` raw format and a clean `label/text` format,
   so no manual renaming needed

The real dataset has 5,572 messages (4,825 ham / 747 spam). My sample has 115. Expect your
real numbers to differ from what's printed below, and probably be more meaningful since
115 hand-written examples is not a real test set.

## Setup

```bash
pip install pandas scikit-learn streamlit
```

## How to run

**1. (Optional) Explore the data first**

```bash
cd src
python explore.py --data ../data/spam.csv
```

Prints class balance, message length stats by class, and the most common words in
spam vs. ham — useful for sanity-checking before you train anything.

**2. Train and evaluate**

```bash
python src/train.py --data data/spam.csv
```

This will:
- Load and clean the text
- Split 80/20 train/test (stratified, so the spam/ham ratio is preserved in both)
- Vectorize with TF-IDF (max 5,000 features)
- Train both Naive Bayes and Logistic Regression
- Print Accuracy, Precision, Recall, and F1 for both
- Save whichever model scores higher on F1, plus the vectorizer, into `models/`

**3. Run the app**

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Paste a message, hit "Check Message," get a
Spam / Not Spam verdict with a confidence score.

## Design notes

- **Why TF-IDF over raw counts**: TF-IDF down-weights words that appear in almost
  every message ("the", "to") relative to words that are distinctive of one class,
  which tends to help linear models like Logistic Regression and gives Naive Bayes
  cleaner signal too.
- **Why compare two models**: Naive Bayes is the textbook baseline for text
  classification and is fast, but Logistic Regression often edges it out once you
  have TF-IDF features. Comparing both (instead of assuming) is the point.
- **Precision vs. Recall tradeoff**: for a spam filter, a false positive (real email
  marked as spam) is usually worse than a false negative (spam that slips through).
  Precision on the "spam" class matters more than Recall in most real deployments.
  Worth keeping in mind if you tune the decision threshold later.
- **Same cleaning function used at training and inference time** (`preprocess.py`
  is imported by both `train.py` and `app.py`). This avoids train/serve skew, a
  common real-world bug where the model is trained on differently-cleaned text
  than what it sees in production.

## Known limitations

- The stop word list is a hand-built ~150 word list, not nltk's. This avoids an
  extra dependency/download but is less exhaustive. Swap in
  `from nltk.corpus import stopwords; stopwords.words('english')` if you want.
- SMS messages are short (avg ~15 words). This pipeline will work on full-length
  emails too, but emails have more structure (headers, HTML, signatures) that
  this simple approach ignores. Treat this as a message classifier, not a
  full email-parsing system.
