
from typing import List
import pandas as pd

def feature_selection(data: pd.DataFrame, selected_feature: List[str]) -> pd.DataFrame:
    """
    Filter the input DataFrame based on the list of selected column names.

    Parameters:
    data (pandas.DataFrame): The raw dataframe to process. The DataFrame includes several feature columns and 'target' column as label.
    selected_feature (List[str]): The list of the selected features.

    Returns:
    pd.DataFrame: The processed dataframe merged target columns.
    """
    # Filter the DataFrame based on the selected features
    processed_data = data[selected_feature + ['target']]

    return processed_data

if __name__ == "__main__":
    # Test the feature_selection function
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'feature3': [7, 8, 9],
        'target': [0, 1, 0]
    })

    selected_features = ['feature1', 'feature3']

    processed_data = feature_selection(data, selected_features)
    print(processed_data)
