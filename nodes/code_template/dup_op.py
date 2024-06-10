import pandas as pd

def dup_op(data:pd.DataFrame,
                      method:str
                      )-> pd.DataFrame:    
    """
    Explanation: Two methods:
    Removing duplicate rows (based on features) entirely from the dataset, keeping only one instance of each unique row, The target column's values in these duplicate rows will not be considered when deciding which rows to keep.
    Replacing the target value of duplicate rows with the median value of the target column. 
    methods: Choose between "drop" and 'median'.

    Parameters:
    data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and 'target' column as label.
    method (str): The method of operation to be applied.
    Returns:
    pd.DataFrame: The new dataframe.
    """
    
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass