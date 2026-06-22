"""
preprocess.py
Cleans raw SMS text: lowercase, strip punctuation/numbers, remove stop words.
"""

import re
import string

# A compact built-in English stop word list so the project has zero
# extra dependencies (no nltk download required). Swap in nltk.corpus.stopwords
# if you prefer a more exhaustive list -- see README for instructions.
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())


def clean_text(text: str) -> str:
    """
    Normalize a raw text message:
      1. lowercase
      2. remove punctuation
      3. remove digits
      4. remove stop words
      5. collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    # strip any remaining non-letter characters (currency symbols, etc.)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens).strip()


if __name__ == "__main__":
    sample = "WINNER!! You have been selected to receive a FREE £1000 cash prize! Call 09061234567 now!"
    print("before:", sample)
    print("after: ", clean_text(sample))
