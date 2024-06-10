import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
def dimensionality_reduction(data:pd.DataFrame,
                      method:str,n_components:int
                      )-> pd.DataFrame:    
    """
    Explanation: Simplify the dataset by reducing the number of features.
    method: Choose between PCA (Principal Component Analysis), UMAP (Uniform Manifold Approximation and Projection ).

    Parameters:
    data (pandas.DataFrame): The input dataset on which dimensionality reduction will be performed. The DataFrame includes several feature columns and 'target' column as label.
    method (str): The method of dimensionality reduction to be applied.
    n_component(int): The number of components (or features) to which the dataset should be reduced.
    Returns:
    pd.DataFrame: The processed dataframe.
    """
    
    pass

if __name__ == "__main__":
    """
    Test all the methods above.
    """
    pass
