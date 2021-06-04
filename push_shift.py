import tqdm
import time
import re
import nltk
import datetime
import pandas as pd

from typing import List, Dict

from psaw import PushshiftAPI

api = PushshiftAPI()
tokenizer = nltk.RegexpTokenizer(r"\w+")

def subreddit_titles(subreddit, quota, sleep=0.5) -> List[str]:
    titles = set()
    gen = api.search_submissions(subreddit=subreddit)

    with tqdm.tqdm(total=quota) as pbar:
        while len(titles) < quota:
            if len(titles) % 500 == 0:
                time.sleep(sleep)
            try:
                titles.add(next(gen).title)
                pbar.update(1)
            except StopIteration:
                # Generator has expired
                break

    return list(set(titles))

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

def remove_stop_words(text: str) -> List[str]:
    # TODO

def lemmatize(text):
    # TODO

def clean_text(text: str) -> List[str]:
    text_no_emoji = remove_emojis(text)
    clean_text = tokenizer.tokenize(text_no_emoji)
    return clean_text


def combine_titles(subreddits: Dict[str, List[str]]) -> pd.DataFrame:
    sub_dfs = []
    for sub, titles in subreddits.items():
        sub_df = pd.DataFrame(titles, columns=["title"])
        sub_df["subreddit"] = sub
        sub_dfs.append(sub_df)
    return pd.concat(sub_dfs)

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

if __name__ == "__main__":
    quota = 10000
    subreddit = "theonion"
    onion = subreddit_titles(subreddit, quota)
    subreddit = "nottheonion"
    not_onion = subreddit_titles(subreddit, quota)
    df = combine_titles({"theonion": onion, "nottheonion": not_onion})
    cleaned_df = clean_titles(df)
    

