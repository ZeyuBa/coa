import pandas as pd

def feature_transformation(
        data:pd.DataFrame,
        methods:str,
        )-> pd.DataFrame:
    """
    Explanation: Modify data features for optimal processing.
    Options: Choose from normalization, standardization.
    
    Parameters:
    data (pandas.DataFrame): The raw dataframe to transform. The DataFrame includes several feature columns and 'target' column as label.
    operation (str): The list of the operations: standardization, normalization, or both.
    Returns:
    pd.DataFrame: The tansformed dataframe.
    """
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass
