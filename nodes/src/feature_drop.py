import pandas as pd
from typing import List

def remove_feat(data: pd.DataFrame, undesired_features: List[str]) -> pd.DataFrame:
    """
    Explanation: Remove undesired features from the given DataFrame.

    Parameters:
    - data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and a 'target' column as a label.
    - undesired_features (List[str]): The columns to be removed.
    
    Returns:
    - pd.DataFrame: The new dataframe with the specified features removed.
    """
    # Ensure that only columns present in the DataFrame are attempted to be removed.
    features_to_remove = [feature for feature in undesired_features if feature in data.columns]
    
    # Drop the undesired features and return the resulting DataFrame.
    return data.drop(columns=features_to_remove)

if __name__ == "__main__":
    # Example test
    # Creating a sample DataFrame
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9],
        'target': [0, 1, 0]
    })
    
    # Defining undesired features to remove
    undesired_features = ['A', 'C']
    
    # Removing the undesired features
    modified_data = remove_feat(data, undesired_features)
    
    # Displaying the result
    print("Original DataFrame:")
    print(data)
    print("\nModified DataFrame (After Removal):")
    print(modified_data)
