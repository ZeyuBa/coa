import pandas as pd
from typing import List
def remove_feat(data:pd.DataFrame,
                      undesired_features: List[str]
                      )-> pd.DataFrame:    
    """
    Explanation: Remove undesired features

    Parameters:
    data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and 'target' column as label.
    undesired_features (List[str]): The columns need to be removed.
    Returns:
    pd.DataFrame: The new dataframe.
    """
    
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass