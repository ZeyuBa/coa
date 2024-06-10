import pandas as pd

def feat_encoder(data:pd.DataFrame,feature_name:str,threshold:float,
                      method:str
                      )-> pd.DataFrame:    
    """
    Explanation: Add a new feature based on clippng the selected feature to a specific value or converting a feature to a boolean variable based on the threshold. 
    methods: Choose between 'clip' and 'bool'.

    Input Parameters:
    data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and 'target' column as label.
    feature_name (str): The name of referenced selected feature.
    method (str): The method of operation to be applied.
    threshold (float): The threshold value used for clipping or boolean conversion.
    Returns:
    pd.DataFrame: The dataframe with new feature.
    """
    
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass