import re
import nltk
import pandas as pd

from typing import List, Dict
from nltk.corpus import stopwords
nltk.download("stopwords")


tokenizer = nltk.RegexpTokenizer(r"\w+")

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
    tokens_without_stop_words = [t for t in text if t not in nltk.stop

def lemmatize(text):
    # TODO

def clean_text(text: str) -> List[str]:
    text_no_emoji = remove_emojis(text)
    clean_text = tokenizer.tokenize(text_no_emoji)
    return clean_text

def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    df.title = df.title.apply(lambda x: clean_text(x))
    return df

# TODO: Get a list of subreddits to use classification on
# TODO: Decide on a simple traditional model first to use
# TODO: Save dataset as pandas dataframe to use later ✓
# TODO: Convert dataset into representation that can be used by the model
# TODO: Decide on preprocessing steps needed
# TODO: Split train test dataset 
# TODO: Decide on two interesting subreddits that could be for this ✓
# TODO: Evaluate performance 
# TODO: Try and use a basic LSTM/network approach
# TODO: Add unit tests for these
