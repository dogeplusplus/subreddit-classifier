import numpy as np
import pandas as pd

from typing import Dict, List
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from itertools import chain

from text_preprocessing import clean_titles

def word_frequencies(df: pd.DataFrame) -> Dict[str, int]:
    df = clean_titles(df)
    word_counts = Counter(chain(*df.title.tolist()))
    return word_counts
