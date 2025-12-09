from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.tree.cart import CART, CARTConfig
from src.tree.gini import gini_gain


# Proof of working classifier
def main():
    df = pd.read_csv("./data/wine-quality.csv")
    x, y = df.iloc[:, :-2], df.iloc[:, -2:-1]

    x = x.to_numpy()
    y = y.to_numpy()

    y = np.where(y <= 5, 0, 1)

    x_train, x_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], train_test_split(x, y, test_size=0.3)
    )

    config = CARTConfig(1000, 2, gini_gain)
    tree = CART(config)
    tree.fit(x_train, y_train)

    y_pred = np.zeros(x_test.shape[0])
    for i, sample in enumerate(x_test):
        y_pred[i] = tree.predict(tree.root, sample)

    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
