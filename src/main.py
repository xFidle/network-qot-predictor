from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.forest.cart import CART, CARTConfig
from src.forest.forest import RandomForest, RandomForestConfig


# Proof of working classifier
def main():
    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 6, 1, 0)

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        train_test_split(x, y, test_size=0.2, random_state=42),
    )

    config = CARTConfig(10, 2)
    tree = CART(config)
    tree.fit(x_train, y_train)

    forest_config = RandomForestConfig(100, config)
    forest = RandomForest(forest_config)
    forest.fit(x_train, y_train)

    sklearn_forest = RandomForestClassifier(100, random_state=42)
    sklearn_forest.fit(x_train, y_train)

    y_pred_tree = tree.predict(x_test)
    y_pred_forest = forest.predict(x_test)

    print("My single tree: ", accuracy_score(y_test, y_pred_tree))
    print("My forest: ", accuracy_score(y_test, y_pred_forest))
    print("Sklearn forest: ", accuracy_score(y_test, sklearn_forest.predict(x_test)))


if __name__ == "__main__":
    main()
