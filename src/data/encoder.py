import pandas as pd


def encode_column_categorical(col: pd.Series) -> pd.Series:
    return col.astype("category").cat.codes


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(encode_column_categorical)


def enocode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, dtype=int)
