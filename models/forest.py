import os
import glob
import mlflow
import pickle
import logging
import argparse
import numpy as np

from typing import List, Tuple, Dict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from text_preprocessing import prepare_dataset

logging.getLogger().setLevel(logging.INFO)

def fetch_logged_data(run_id: str) -> Tuple[Dict, Dict, Dict, Dict]:
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

def grid_train(X: np.array, y: np.array) -> RandomForestClassifier:
    clf = RandomForestClassifier()

    param_grid = {
        "n_estimators": [20, 100],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=3)
    grid_search.fit(X, y)

    return grid_search.best_estimator_


def parse_arguments():
    parser = argparse.ArgumentParser("Grid train forest classifier")
    parser.add_argument("-f", "--files", nargs="+", type=str, help="Subreddit CSV files to use", default=glob.glob("data/*.csv"))
    parser.add_argument("-n", "--name", type=str, help="Name of model to save as")
    parser.add_argument("-s", "--splits", nargs=2, type=float, help="Validation and test dataset ratios", default=[0.2, 0.1])

    args = parser.parse_args()
    return args


def main(args):
    splits = args.splits
    dataset = prepare_dataset(args.files, splits)

    with mlflow.start_run() as run:
        best_forest = grid_train(dataset["train"]["X"], dataset["train"]["y"])

        for subset in ("train", "validation", "test"):
            predictions = best_forest.predict(dataset[subset]["X"])
            accuracy = accuracy_score(dataset[subset]["y"], predictions)

            predicted_class = np.argmax(predictions, axis=-1)
            labels = np.argmax(dataset[subset]["y"], axis=-1)
            f1 = f1_score(labels, predicted_class)
            conf_matrix = confusion_matrix(labels, predicted_class)

            mlflow.log_metric(f"{subset}_accuracy", accuracy)
            mlflow.log_metric(f"{subset}_f1", f1)

        mlflow.log_param("labels", labels)
        mlflow.log_params(best_forest.get_params())
        mlflow.sklearn.log_model(best_forest, "sk_models")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
