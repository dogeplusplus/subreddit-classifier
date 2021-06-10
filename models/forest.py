import os
import glob
import pickle
import logging
import argparse
import numpy as np

from typing import List
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from text_preprocessing import prepare_dataset

logging.getLogger().setLevel(logging.INFO)

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


def save_model(model: RandomForestClassifier, name: str):
    idx = 0
    os.makedirs("model_instances", exist_ok=True)
    model_path = f"model_instances/{name}_{idx}.pkl"
    while os.path.isfile(model_path):
        idx += 1
        model_path = f"model_instances/{name}_{idx}.pkl"

    output = open(model_path, "wb")
    pickle.dump(model, output)
    logging.info(f"Saved model to: {model_path}")


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

    best_forest = grid_train(dataset["train"]["X"], dataset["train"]["y"])

    for subset in ("train", "validation", "test"):
        predictions = best_forest.predict(dataset[subset]["X"])
        accuracy = accuracy_score(dataset[subset]["y"], predictions)

        predicted_class = np.argmax(predictions, axis=-1)
        labels = np.argmax(dataset[subset]["y"], axis=-1)
        f1 = f1_score(labels, predicted_class)
        conf_matrix = confusion_matrix(labels, predicted_class)

        logging.info(f"{subset}{'-' * (20 - len(subset))}")
        logging.info(f"accuracy: {accuracy:.5f}")
        logging.info(f"F1 score: {f1:.5f}")
        logging.info(f"Classes: {dataset['labels']}")
        logging.info(conf_matrix)

    save_model(best_forest, args.name)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
