# coding: utf-8

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import umap


def test_umap_digit():
    # sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

    digits = load_digits()
    print("loaded digits images")

    digits_df = pd.DataFrame(digits.data[:, 1:11])
    print("digits df: ", digits_df)
    digits_df["digit"] = pd.Series(digits.target).map(lambda x: "Digit {}".format(x))
    print("digits df: ", digits_df)

    reducer = umap.UMAP(random_state=42)
    print(
        f"digits.data: type: {type(digits.data)}, shape: {digits.data.shape}, max: {np.max(digits.data)}, min: {np.min(digits.data)}"
    )
    reducer.fit(digits.data)

    embedding = reducer.transform(digits.data)
    embedding.shape

    print([1, 2] + [3, 4])

