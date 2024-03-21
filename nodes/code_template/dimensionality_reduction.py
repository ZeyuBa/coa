import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
def dimensionality_reduction(data:pd.DataFrame,
                      methods:str,n_components:int
                      )-> pd.DataFrame:    
    """
    Explanation: Simplify the dataset by reducing the number of features.
    methods: Choose between PCA (Principal Component Analysis), ISOMAP (Isometric Mapping).

    Parameters:
    data (pandas.DataFrame): The input dataset on which dimensionality reduction will be performed. The DataFrame includes several feature columns and 'target' column as label.
    methods (str): The method of dimensionality reduction to be applied.
    n_component(int): The number of components (or features) to which the dataset should be reduced.
    pd.DataFrame: The processed dataframe.
    """
    
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass
