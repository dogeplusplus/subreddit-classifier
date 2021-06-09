from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from text_preprocessing import prepare_dataset

def train(data_paths):
    clf = RandomForestClassifier()

    param_grid = {
        "n_estimators": [20, 100, 500],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
    }

    dataset = prepare_dataset(
    grid_search = GridSearchCV(X)

