import os
import tqdm
import  time
import logging
import argparse
import pandas as pd

from typing import List, Dict
from psaw import PushshiftAPI

api = PushshiftAPI()

logger = logging.getLogger(__name__)

def subreddit_titles(subreddit: str, quota: int, sleep=0.5) -> List[str]:
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


def combine_titles(subreddits: Dict[str, List[str]]) -> pd.DataFrame:
    sub_dfs = []
    for sub, titles in subreddits.items():
        sub_df = pd.DataFrame(titles, columns=["title"])
        sub_df["subreddit"] = sub
        sub_dfs.append(sub_df)
    return pd.concat(sub_dfs)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Obtain subreddit titles using PSAW")
    parser.add_argument("-s", "--subreddits", nargs="+", type=str, help="Subreddits to get titles from")
    parser.add_argument("-q", "--quota", type=int, help="Number of titles to scrape per subreddit", default=1000)

    args = parser.parse_args()
    return args


def main(args):
    subreddits = args.subreddits
    quota = args.quota
    os.makedirs("data", exist_ok=True)

    logger.info(f"Downloading data for {subreddits}")

    for subreddit in subreddits:
        destination = f"data/{subreddit}_{quota}.csv"
        if os.path.isfile(destination):
            logger.info(f"{quota} titles from {subreddit} already exists. Skipping...")
        else:
            titles = subreddit_titles(subreddit, quota)
            df = pd.DataFrame({"title": titles, "subreddit": [subreddit] * len(titles)})
            df.to_csv(destination, index=False)
            logger.info("Saved {quota} titles from {subreddit} to {destination}")

    logger.info("Finished downloading data from Reddit.")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

