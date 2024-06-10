
import stat
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

def dimensionality_reduction(data: pd.DataFrame, methods: str, n_components: int) -> pd.DataFrame:
    """
    Explanation: Simplify the dataset by reducing the number of features.
    methods: Choose between PCA (Principal Component Analysis), ISOMAP (Isometric Mapping).

    Parameters:
    data (pandas.DataFrame): The input dataset on which dimensionality reduction will be performed. The DataFrame includes several feature columns and 'target' column as label.
    method (str): The method of dimensionality reduction to be applied.
    n_components (int): The number of components (or features) to which the dataset should be reduced.
    Returns:
    pd.DataFrame: The processed dataframe.
    """
    if methods == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data.drop('target', axis=1))
    elif methods == 'ISOMAP':
        isomap = Isomap(n_neighbors=n_components, n_components=2)
        reduced_data = isomap.fit_transform(data.drop('target', axis=1))
    else:
        raise ValueError("Invalid method. Choose between 'PCA' and 'ISOMAP'.")

    reduced_df = pd.DataFrame(reduced_data, columns=[f'component_{i+1}' for i in range(reduced_data.shape[1])])
    reduced_df['target'] = data['target']
    
    return reduced_df

if __name__ == "__main__":
    # Create a sample dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8],
        'target': [0, 1, 0, 1]
    })
    data=pd.read_pickle('data.pkl')
    print(data.skew().sort_values(key=lambda x: abs(x),ascending=False).to_string())
    # import time
    # import warnings
    # warnings.filterwarnings("ignore")
    # start=time.time()
    # # Test PCA method
    # pca_reduced_data = dimensionality_reduction(data, 'PCA', 6)
    # print("PCA Reduced Data:")
    # print(pca_reduced_data)
    # print('data time cost:',time.time()-start)
    # start=time.time()
    # # Test ISOMAP method
    # isomap_reduced_data = dimensionality_reduction(data, 'ISOMAP', 6)
    # print("\nISOMAP Reduced Data:")
    # print(isomap_reduced_data)
    # print('data time cost:',time.time()-start)