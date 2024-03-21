
from typing import List
import pandas as pd
from sklearn.utils import resample

def resampling(data: pd.DataFrame, methods: str) -> pd.DataFrame:
    """
    Adjust the balance of classes within a dataset by either upsampling or downsampling.
    Explanation: Balance the data by increasing the representation of the minority class or decreasing the majority class.
    method: Choose between upsampling or downsampling.

    Parameters:
    data (pandas.DataFrame): The raw dataframe to resample. The DataFrame includes several feature columns and 'target' column as label.
    method (str): upsampling or downsampling. Upsampling increases the number of instances in the minority class to address imbalance, typically using methods like SMOTE (Synthetic Minority Over-sampling Technique). Downsampling reduces the number of instances in the majority class, making the class distribution more balanced.
    
    Returns:
    pd.DataFrame: The processed dataframe merged target columns.
    """
    
    if methods == 'upsampling':
        # Upsample the minority class
        minority_class = data[data['target'] == 1]
        majority_class = data[data['target'] == 0]
        minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        data_resampled = pd.concat([majority_class, minority_upsampled])
    elif methods == 'downsampling':
        # Downsample the majority class
        minority_class = data[data['target'] == 1]
        majority_class = data[data['target'] == 0]
        majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
        data_resampled = pd.concat([majority_downsampled, minority_class])
    else:
        raise ValueError("Invalid method. Choose between 'upsampling' or 'downsampling'.")

    return data_resampled

if __name__ == "__main__":
    # Test the resampling function
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 20, 30, 40, 50, 60],
        'target': [0, 0, 0, 1, 1, 1]
    })
    
    print("Original data:")
    print(data)
    data=pd.read_pickle('data.pkl')
    
    upsampled_data = resampling(data, 'upsampling')
    print("\nUpsampled data:")
    print(upsampled_data)
    
    downsampled_data = resampling(data, 'downsampling')
    print("\nDownsampled data:")
    print(downsampled_data)
