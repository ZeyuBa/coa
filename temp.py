# import os
# from openai import OpenAI
# prompt_template="{message}"
# def openai_node(model_config,prompt_config):
#     client=OpenAI(api_key="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",)
#     if model_config['model_name_or_path']=='gpt-4':
#         model="gpt-4-1106-preview"
#     else: model='gpt-3.5-turbo'
#     model_kwargs=model_config['model_kwargs']
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt_template.format(**prompt_config),
#             }
#         ],
#         model=model,**model_kwargs
#     )
#     print(chat_completion)
#     return chat_completion.choices[0].message.content

# model_config={"model_name_or_path":'gpt-3.5-turbo',"model_kwargs":{"temperature": 0}}
# prompt_config={'message':"hello"}
# print(openai_node(model_config,prompt_config))
import pandas as pd
import numpy as np

def generate_test_data(num_rows, num_features, num_categories):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random numerical data
    numerical_data = np.random.randn(num_rows, num_features)
    
    # Generate random categorical data
    categories = [(i) for i in range(1, num_categories + 1)]
    categorical_data = np.random.choice(categories, size=num_rows)
    
    # Generate a target variable with some noise
    coefficients = np.random.rand(num_features)
    target = numerical_data @ coefficients + np.random.randn(num_rows) * 0.5  # Adding some noise
    
    # Create DataFrame for numerical data
    df = pd.DataFrame(numerical_data, columns=[f'Feature_{i+1}' for i in range(num_features)])
    i=len(df.columns)
    df[f'Feature_{i+1}'] = categorical_data
    df['Target'] = target
    
    return df
import pandas as pd
import numpy as np

def generate_classification_data(num_rows, num_features, num_categories, num_classes):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random numerical data
    numerical_data = np.random.randn(num_rows, num_features)
    
    # Generate random categorical data
    categories = [(i) for i in range(1, num_categories + 1)]
    categorical_data = np.random.choice(categories, size=num_rows)
    
    # Generate random class labels for the target
    target_labels = np.random.randint(0, num_classes, size=num_rows)
    
    # Create DataFrame for numerical data
    df = pd.DataFrame(numerical_data, columns=[f'Feature_{i+1}' for i in range(num_features)])
    i=len(df.columns)
    df[f'Feature_{i+1}'] = categorical_data
    df['Target'] = target_labels
    
    return df

# Example usage
df = generate_classification_data(num_rows=100, num_features=5, num_categories=3, num_classes=4)
df.to_pickle('./classification_test.pkl')

# Example usage
df = generate_test_data(num_rows=100, num_features=5, num_categories=3)
df.to_pickle('./regression_test.pkl')
print(df.head())
