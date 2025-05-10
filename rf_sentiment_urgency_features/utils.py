\
import re
import pandas as pd # type: ignore
from tqdm import tqdm # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import logging
from pandarallel import pandarallel # type: ignore

# Initialize pandarallel, suppressing output as it's a utility module
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    pandarallel.initialize(progress_bar=False)

log = logging.getLogger(__name__)

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\\S+", "", s)
    s = re.sub(r"[^a-záéíóúñü0-9\\s]", "", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def extract_features(df, max_feats):
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("pandarallel").setLevel(logging.ERROR)
    logging.getLogger("sklearn").setLevel(logging.ERROR)
    vectorizer = TfidfVectorizer(
        max_features=max_feats, ngram_range=(1, 2), min_df=1, max_df=1.0
    )
    try:
        X_text = vectorizer.fit_transform(df["cleaned"])
    except ValueError as e:
        log.warning(f"[ERROR] {e}\\nTrying with even more permissive parameters...")
        vectorizer = TfidfVectorizer(max_features=max_feats, ngram_range=(1, 1), min_df=1, max_df=1.0)
        X_text = vectorizer.fit_transform(df["cleaned"])
    return X_text, vectorizer

def compute_features(df):
    tqdm.pandas(desc="Limpieza de texto")
    df["cleaned"] = df["tweet_text"].progress_apply(clean_text)
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["cleaned"].progress_apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["sentiment"] = (df["sentiment"] + 1) / 2
    keywords = ["ayuda", "sos", "auxilio"]
    df["urgency"] = df["cleaned"].progress_apply(lambda t: int(any(k in t for k in keywords)))
    df["tweet_len"] = df["cleaned"].progress_apply(len)
    df["num_exclaims"] = df["cleaned"].progress_apply(lambda x: x.count("!"))
    df["num_questions"] = df["cleaned"].progress_apply(lambda x: x.count("?"))
    df["num_hashtags"] = df["tweet_text"].progress_apply(lambda x: len(re.findall(r"#\\w+", str(x))))
    df["num_mentions"] = df["tweet_text"].progress_apply(lambda x: len(re.findall(r"@\\w+", str(x))))
    for col in ["Tweet_Likes", "Tweet_Retweets"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    return df
