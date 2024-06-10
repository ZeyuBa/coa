import pandas as pd

def dup_op(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Explanation: Two methods:
    1. Removing duplicate rows (based on features) entirely from the dataset, keeping only one instance of each unique row. 
       The target column's values in these duplicate rows will not be considered when deciding which rows to keep.
    2. Replacing the target value of duplicate rows with the median value of the target column.
    
    Parameters:
    data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and 'target' column as label.
    method (str): The method of operation to be applied. Choose between "drop" and "median".
    
    Returns:
    pd.DataFrame: The new dataframe.
    """
    
    if method == "drop":
        # Dropping duplicates based on all columns except 'target'
        new_data = data.drop_duplicates(subset=data.columns.difference(['target']), keep='first')
    elif method == "median":
        # Calculating median of 'target' for duplicates and replacing the 'target' value for all duplicates
        new_data = data.copy()
        for _, subset in data[data.duplicated(subset=data.columns.difference(['target']), keep=False)].groupby(list(data.columns.difference(['target']))):
            median_target = subset['target'].median()
            new_data.loc[subset.index, 'target'] = median_target
    else:
        raise ValueError("Invalid method. Choose between 'drop' and 'median'.")
    
    return new_data

if __name__ == "__main__":
    # Example DataFrame
    df = pd.DataFrame({
        'feature1': [1, 2, 2, 3],
        'feature2': ['A', 'B', 'B', 'C'],
        'target': [100, 200, 300, 300]
    })

    print("Original DataFrame:")
    print(df)

    # Test "drop" method
    print("\nDataFrame after 'drop':")
    print(dup_op(df, "drop"))

    # Test "median" method
    print("\nDataFrame after 'median':")
    print(dup_op(df, "median"))
