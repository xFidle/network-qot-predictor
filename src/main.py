from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.forest.cart import CART, CARTConfig
from src.forest.forest import RandomForest, RandomForestConfig
from src.selector.selector import DiversitySelector, UncertaintySelector


# Proof of working classifier
def main():
    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 6, 1, 0)

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.5)
    )

    config = CARTConfig(10, 2)

    tree = CART(config)
    tree.fit(x_train, y_train)

    forest_config = RandomForestConfig(100, config)
    forest = RandomForest(forest_config)
    forest.fit(x_train, y_train)

    sklearn_forest = RandomForestClassifier(100, random_state=42)
    sklearn_forest.fit(x_train, y_train)

    diverisity_selector = DiversitySelector()
    uncertainty_selector = UncertaintySelector()

    different_samples = diverisity_selector(forest, x_test)
    uncertain_samples = uncertainty_selector(forest, x_test)

    print("Most different:\n", different_samples)
    print("Mose unsure:\n", uncertain_samples)

    print("My single tree: ", accuracy_score(y_test, tree.predict(x_test)))
    print("My forest: ", accuracy_score(y_test, forest.predict(x_test)))
    print("SKLEARN forest: ", accuracy_score(y_test, sklearn_forest.predict(x_test)))


if __name__ == "__main__":
    main()
