import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

def get_feature_specs(df):
    """
    Extracts feature specifications from an existing DataFrame.
    
    Args:
    - df (pd.DataFrame): The DataFrame from which to extract feature specifications.
    
    Returns:
    - list of tuples: Each tuple contains ('column_name', 'data_type'), where 'data_type' is either 'numeric' or 'categorical'.
    """
    feature_specs = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_specs.append((column, 'numeric'))
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            feature_specs.append((column, 'categorical'))
    return feature_specs



def generate_test_data(num_rows, feature_specs, num_classes=None):
    """
    Generates test data for regression or classification tasks based on specified feature columns.
    
    Args:
    - num_rows (int): Number of rows in the DataFrame.
    - feature_specs (list of tuples): Each tuple contains ('column_name', 'data_type'),
      where 'data_type' is either 'numeric' or 'categorical'.
    - num_classes (int, optional): Number of classes for classification tasks. If None, generates data for regression.

    Returns:
    - DataFrame: Generated test data with target column.
    """
    np.random.seed(42)  # For reproducibility
    data = {}
    
    # Generate data based on feature specifications
    for column_name, data_type in feature_specs:
        if data_type == 'numeric':
            data[column_name] = np.random.randn(num_rows)
        elif data_type == 'categorical':
            categories = ['Category_A', 'Category_B', 'Category_C']
            data[column_name] = np.random.choice(categories, num_rows)

    df = pd.DataFrame(data)

    # Generate target variable based on the task type
    if num_classes is None:
        # Regression task: target as a linear combination of numerics with noise
        numeric_cols = [name for name, dtype in feature_specs if dtype == 'numeric']
        coefficients = np.random.rand(len(numeric_cols))
        df['Target'] = df[numeric_cols].dot(coefficients) + np.random.randn(num_rows) * 0.5
    else:
        # Classification task: target as random classes
        df['Target'] = np.random.randint(0, num_classes, num_rows)
    
    return df
if __name__=='__main__':
    # Example usage
    feature_specs = [('Feature_1', 'numeric'), ('Feature_2', 'numeric'), ('Feature_3', 'categorical')]
    regression_df = generate_test_data(num_rows=100, feature_specs=feature_specs)
    classification_df = generate_test_data(num_rows=100, feature_specs=feature_specs, num_classes=3)

