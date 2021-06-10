import re
import nltk
import spacy
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
nlp = spacy.load('en_core_web_sm')

tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = nltk.WordNetLemmatizer()

def remove_emojis(text: str) -> List[str]:
    regex = re.compile(
        pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+",
        flags=re.UNICODE
    )
    return regex.sub(r"", text)

def remove_stop_words(text: List[str]) -> List[str]:
    tokens_without_stop_words = [t for t in text if t not in stopwords.words()]
    return tokens_without_stop_words


def preprocess_title(text: str) -> List[str]:
    no_emoji = remove_emojis(text)
    tokens = tokenizer.tokenize(no_emoji)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    stopless = remove_stop_words(lemmas)

    return stopless

def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    df.title = df.title.apply(lambda x: preprocess_title(x))
    return df

def sentence_embedding(sentences: pd.Series) -> np.array:
    embeddings = np.array([nlp(sen).vector for sen in sentences])
    return embeddings

def split_train_test_validation(X: np.array, y: np.array, ratios: Tuple[int, int]) -> List[np.array]:
    valid_split, test_split = ratios

    assert sum(ratios) < 1, "Ratios to split are not valid"
    assert valid_split > 0, "Validation size must be more than 0"
    assert test_split > 0, "Test size must be more than 0"

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size = valid_split + test_split)

    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, test_size=test_split / (valid_split + test_split))

    return [X_train, X_valid, X_test, y_train, y_valid, y_test]

def prepare_dataset(data_paths: List[str], split_ratios: Tuple[int, int, int]) -> Dict[str, Dict[str, np.array]]:
    df = pd.concat([pd.read_csv(csv) for csv in data_paths])
    X = sentence_embedding(df.title)
    y = pd.get_dummies(df.subreddit)

    labels = y.columns
    y = y.to_numpy()

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_train_test_validation(X, y, split_ratios)

    dataset = {
        "train": {"X": X_train, "y": y_train},
        "validation": {"X": X_valid, "y": y_valid},
        "test": {"X": X_test, "y": y_test},
        "labels": labels,
    }

    return dataset

