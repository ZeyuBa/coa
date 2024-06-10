from typing import List
import pandas as pd
from sklearn.utils import resample

def resampling(
        data:pd.DataFrame,
        method:str,
)-> pd.DataFrame:
    """
    Adjust the balance of classes within a dataset by either upsampling or downsampling.
    Explanation: Balance the data by increasing the representation of the minority class or decreasing the majority class.
    method: Choose between upsampling or downsampling.

    Parameters:
    data (pandas.DataFrame): The raw dataframe to resample. The DataFrame includes several feature columns and 'target' column as label.
    method (str): upsampling or downsampling. Upsampling increases the number of instances in the minority class to address imbalance, typically using methods like SMOTE (Synthetic Minority Over-sampling Technique). Downsampling reduces the number of instances in the majority class, making the class distribution more balanced.
    Returns:
    pd.DataFrame:  The processed dataframe merged target columns.
    """

    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass
