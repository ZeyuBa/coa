import pandas as pd
def feat_encoder(data: pd.DataFrame, feature_name: str, threshold: float, method: str) -> pd.DataFrame:
    """
    Explanation: Add a new feature based on clipping the selected feature to a specific value or converting a feature to a boolean variable based on the threshold. 
    Methods: Choose between 'clip' and 'bool'.

    Parameters:
    data (pandas.DataFrame): The input dataset on which operation will be performed. The DataFrame includes several feature columns and 'target' column as label.
    feature_name (str): The name of the referenced selected feature.
    method (str): The method of operation to be applied.
    threshold (float): The threshold value used for clipping or boolean conversion.
    Returns:
    pd.DataFrame: The dataframe with the new feature.
    """
    
    if method == 'clip':
        # Create a new feature by clipping the selected feature to the threshold
        new_feature_name = f"{feature_name}_clipped"
        data[new_feature_name] = data[feature_name].clip(None,upper=threshold)
    elif method == 'bool':
        # Create a new feature by converting the selected feature to boolean based on the threshold
        new_feature_name = f"{feature_name}_bool"
        data[new_feature_name] = data[feature_name] > threshold
    else:
        raise ValueError("Method not recognized. Choose between 'clip' and 'bool'.")

    return data

if __name__ == "__main__":
    # Example data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 15, 20, 25, 30, 35],
        "target": [0,1,0,1,0,0]
    })
    # Testing the 'clip' method
    data_with_clipped_feature = feat_encoder(data.copy(), 'feature2', 25, 'clip')
    print("Data with clipped feature:\n", data_with_clipped_feature)
    
    # Testing the 'bool' method
    data_with_bool_feature = feat_encoder(data.copy(), 'feature2', 20, 'bool')
    print("\nData with boolean feature:\n", data_with_bool_feature)
