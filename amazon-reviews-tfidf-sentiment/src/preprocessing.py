import pandas as pd

def load_dataset(path, nrows=None):
    df = pd.read_csv(path, sep="\t", header=None, names=["text"], nrows=nrows)
    df["label"] = df["text"].str.split(" ",n=1).str[0]
    df["review"] = df["text"].str.split(" ",n=1).str[1]
    df["label"] = df["label"].map({"__label__1": 0, "__label__2": 1})

    return df[["review", "label"]]

#### Preprocess Review sentiments

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
        
    return " ".join(tokens)

# Add "clear_review" column to train/test_df

if "clean_review" not in train_df.columns:
    train_df["clean_review"] = train_df["review"].astype(str).apply(preprocess_text)
    
if "clean_review" not in test_df.columns:
    test_df["clean_review"] = test_df["review"].astype(str).apply(preprocess_text)