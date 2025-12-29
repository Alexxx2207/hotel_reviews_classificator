from pathlib import Path
import kagglehub
import pandas as pd
from tripadvisor_dataset_settings import (
    DATASET_FILE_NAME, DATASET_NAME, NEG_RATINGS, POS_RATINGS, TEXT_COL, RATING_COL
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


# def split_train_val_test(df: pd.DataFrame):
    

def main():
    df = load_dataset_from_csv()
    df = to_binary_labels

if __name__ == "__main__":
    main()