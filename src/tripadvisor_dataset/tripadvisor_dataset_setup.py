from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from tripadvisor_dataset_settings import (
    DATASET_FILE_NAME, DATASET_NAME, NEG_RATINGS, POS_RATINGS, RANDOM_STATE, TEST_SIZE, TEXT_COL, RATING_COL
) 


def load_dataset_from_csv() -> pd.DataFrame:
    """Loads data from a csv file in a dataframe table. 

    Returns:
        pd.DataFrame: csv data in the dataframe
    """
    
    dataset_dir = kagglehub.dataset_download(DATASET_NAME)

    df = pd.read_csv(Path(dataset_dir) / DATASET_FILE_NAME)
    df = df[[TEXT_COL, RATING_COL]].dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df[RATING_COL] = df[RATING_COL].astype(int)
    
    return df


def to_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Substitute ratings with binary values for positive and negative review

    Args:
        df (pd.DataFrame): the data in a dataframe object

    Returns:
        pd.DataFrame: new dataframe object with applied substitution
    """
    
    df = df[df[RATING_COL].isin(NEG_RATINGS | POS_RATINGS)].copy()
    
    df["label"] = df[RATING_COL].apply(lambda r: 0 if r in NEG_RATINGS else 1)
    
    return df[[TEXT_COL, "label"]]


def split_train_test(df: pd.DataFrame):
    """Split dataset in train and test data

    Args:
        df (pd.DataFrame): data to split

    Returns:
        _type_: tuple of train,test dataset split
    """
    
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COL], df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    return (x_train, y_train), (x_test, y_test)


def save_splits(train, test):
    """Saves the splits on the disk

    Args:
        train (_type_): train dataframe
        test (_type_): test dataframe
    """
    project_dir = Path(__file__).resolve().parents[1]
    processed_data_path = project_dir / "data"
    processed_data_path.mkdir(parents=True, exist_ok=True)

    (X_train, y_train) = train
    (X_test, y_test) = test

    pd.DataFrame({TEXT_COL: X_train, "label": y_train}).to_csv(processed_data_path / "train.csv", index=False)
    pd.DataFrame({TEXT_COL: X_test, "label": y_test}).to_csv(processed_data_path / "test.csv", index=False)


def main():
    df = load_dataset_from_csv()
    df = to_binary_labels(df)
    
    train, test = split_train_test(df)
    save_splits(train, test)


if __name__ == "__main__":
    main()