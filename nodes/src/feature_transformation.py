
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def feature_transformation(data: pd.DataFrame, methods: str) -> pd.DataFrame:
    """
    Explanation: Modify data features for optimal processing.
    Options: Choose from normalization, standardization.

    Parameters:
    data (pandas.DataFrame): The raw dataframe to transform. The DataFrame includes several feature columns and 'target' column as label.
    operation (str): The list of the operations: standardization, normalization, or both.
    Returns:
    pd.DataFrame: The transformed dataframe.
    """
    if methods == 'standardization':
        scaler = StandardScaler()
        data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    elif methods == 'normalization':
        scaler = MinMaxScaler()
        data[data.columns[:-1]] = scaler.fit_transform(data[data.columns[:-1]])
    elif methods == 'both':
        standard_scaler = StandardScaler()
        data[data.columns[:-1]] = standard_scaler.fit_transform(data[data.columns[:-1]])
        
        minmax_scaler = MinMaxScaler()
        data[data.columns[:-1]] = minmax_scaler.fit_transform(data[data.columns[:-1]])
    
    return data

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    # Create a sample dataframe for testing
    data = pd.read_pickle('data.pkl')
    
    print("Original Data:")
    print(data)

    transformed_data_standardization = feature_transformation(data.copy(), 'standardization')
    print("\nData after standardization:")
    print(transformed_data_standardization)
    
    transformed_data_normalization = feature_transformation(data.copy(), 'normalization')
    print("\nData after normalization:")
    print(transformed_data_normalization)

    transformed_data_both = feature_transformation(data.copy(), 'both')
    print("\nData after both standardization and normalization:")
    print(transformed_data_both)
