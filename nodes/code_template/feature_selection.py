from typing import List
import pandas as pd
def feature_selection(data:pd.DataFrame,
                      selected_feature:List[str],
                      )-> pd.DataFrame:    
    """
    Filter the input DataFrame based on the list of selected column names.

    Parameters:
    data (pandas.DataFrame): The raw dataframe to process. The DataFrame includes several feature columns and 'target' column as label.s
    selected_feature (List[str]): The list of the selected features.
    Returns:
    pd.DataFrame: The processed dataframe merged target columns.
    """

    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass