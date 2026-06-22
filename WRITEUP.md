# Write-up: Spam Classifier

## What I built

A pipeline that cleans SMS text, converts it to TF-IDF features, and trains both
a Naive Bayes and a Logistic Regression model to classify messages as spam or ham.
The better model (by F1 score) gets saved and served through a Streamlit app where
you can paste a message and get a live prediction with a confidence score.

## What worked

- **TF-IDF + Naive Bayes was a strong, fast baseline**, exactly as advertised in
  every text classification tutorial. It trained in well under a second even with
  a few thousand rows, and the cleaning step (lowercase, strip punctuation/digits,
  remove stop words) made a visible difference in which words the model latched onto.
- **The EDA step caught something useful early**: spam messages are noticeably
  longer than ham messages (in my sample, roughly double the word count), and the
  most common words in spam were almost cartoonishly predictable — "free," "prize,"
  "win," "call," "cash," "txt." This is reassuring: it means the signal is real and
  the model isn't going to need anything fancy to find it.
- **Reusing the same `clean_text()` function in both training and the app** avoided
  a bug I've hit before, where text gets cleaned slightly differently at inference
  time than at training time and the model quietly underperforms in production.
  Worth calling out because it's an easy thing to get wrong and easy to miss when
  testing.

## What didn't work / what surprised me

- On a small sample, Naive Bayes and Logistic Regression came out essentially tied
  on every metric. I expected at least a small gap. My read is that the dataset's
  signal is strong enough (spam words are very distinctive) that the choice of
  model barely matters here — most of the heavy lifting is done by the TF-IDF
  features, not the classifier itself. With the full ~5,500-message dataset I'd
  expect to see more separation, especially in Recall, since Logistic Regression
  tends to handle the class imbalance (ham massively outnumbers spam in the real
  dataset) a bit more gracefully than Naive Bayes' independence assumption.
- The £ currency symbol and similar non-ASCII characters needed an extra regex
  pass beyond just `string.punctuation` — Python's punctuation set doesn't cover
  currency symbols, so without the extra cleanup step they were leaking into the
  vocabulary as junk tokens.
- I went with a hand-written stop word list instead of nltk's, mainly to avoid an
  extra dependency and a download step that can fail in restricted environments.
  It's "good enough" but not as exhaustive as nltk's — something I'd swap if this
  were going further than a learning project.

## What I'd do differently next time

- **Try character n-grams in addition to word-level TF-IDF.** Spam messages often
  use deliberate misspellings ("fr33", "w1n") specifically to dodge keyword filters.
  Word-level TF-IDF misses these; character n-grams would catch them.
- **Tune the classification threshold instead of using the default 0.5.** Since
  false positives (flagging a real message as spam) are usually more costly than
  false negatives in a real spam filter, I'd push the threshold to favor precision
  on the spam class, and make that tradeoff explicit and adjustable rather than
  hard-coded.
- **Add a confusion matrix to the evaluation output**, not just the four summary
  metrics. Accuracy/Precision/Recall/F1 tell you the rates, but seeing the actual
  counts (especially false positives) gives a more visceral sense of what the model
  gets wrong and on what kind of messages.
- **Test on genuinely new, out-of-distribution text** — messages that don't sound
  like SMS spam at all (e.g. phishing emails, which are longer and more "professional"
  sounding than SMS spam). The SMS Spam Collection dataset is from 2011-era UK SMS
  text, so a model trained purely on it may not generalize well to modern email
  phishing, which has gotten more sophisticated and less obviously spammy-sounding.
