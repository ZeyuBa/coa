### Import following necessary libs
from os import pipe
from typing import List,Tuple
import pandas as pd
from dimensionality_reduction import dimensionality_reduction
from feature_selection import feature_selection
from feature_transformation import feature_transformation
from resampling import resampling
import xgboost as xgb
xgb.set_config(verbosity=0)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import pickle
import math
import demjson3
from pathlib import Path
root_dir=Path(__file__).parent.resolve()
save_path=root_dir
# print('#####',save_path)
def xgboost_model(type, params):
    if type == 'classification':
        return xgb.XGBClassifier(**params, eval_metric='mlogloss')
    elif type == 'regression':
        return xgb.XGBRegressor(**params,eval_metric='rmse')
    else:
        raise ValueError("Invalid task type. Please choose either 'classification' or 'regression'.")


def evaluate_model(model, X_test, y_test, type):
    if type == 'classification':
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    elif type == 'regression':
        y_pred = model.predict(X_test)
        return math.sqrt(mean_squared_error(y_test, y_pred))
    else:
        raise ValueError("Invalid task type. Please choose either 'classification' or 'regression'.")

def main(X_path,y_path, params_path, mode):
    info_dict=demjson3.decode_file(params_path)
    type,params=info_dict['type'],info_dict['params']
    X=pd.read_pickle(X_path)
    y=pd.read_pickle(y_path)
    if mode=='predict':
                # To load the model later
        model = xgb.Booster()
        model.load_model('best_model.json')
        dtest = xgb.DMatrix(X)

        # Now use dtest for prediction
        y_pred = model.predict(dtest)
        df_y_pred = pd.DataFrame({'Strength':y_pred})
        df_y_pred.to_csv('prediction.csv', index=False)
        return {"test":-1.0}, None,None
    else:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        kf = KFold(n_splits=5)
        traning_log = {}
        metrics = {}
        best_model = None
        best_metric = float('inf')  # or -float('inf') depending on whether you want to maximize or minimize the metric
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):  # Change X to X_train
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]  # Change X to X_train
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]  # Change y to y_train

            model = xgboost_model(type=type, params=params)
            model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)

            traning_log[f'fold_{fold}'] = model.evals_result()
            fold_metric = evaluate_model(model, X_fold_val, y_fold_val, type)  # Assuming evaluate_model returns a single metric value
            metrics[f'fold_{fold}'] = fold_metric
            if type == 'regression':
                if fold_metric < best_metric:
                    best_metric = fold_metric
                    best_model = model
            else:
                if fold_metric > best_metric:
                    best_metric = fold_metric
                    best_model = model
                # Save the best model

        metrics['test']=evaluate_model(best_model, X_test, y_test, type)

        return metrics, traning_log,best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run machine learning model training and evaluation.')
    parser.add_argument('X_path', type=str,help='Path to the feature file')
    parser.add_argument('y_path', type=str,help='Path to the label file')
    parser.add_argument('params_path', type=str ,help='Path to the model configure file')
    parser.add_argument('mode', type=str ,help='train or predict')
    args = parser.parse_args()

    metrics, training_log,best_model=main(args.X_path,args.y_path, args.params_path, args.mode)

    demjson3.encode_to_file('result.json', {'metrics': metrics, 'training_log': training_log},overwrite=True)
    import os 
    import demjson3
    if args.mode == 'train':
        if not os.path.exists('best_metric.json'):
            demjson3.encode_to_file('best_metric.json',metrics['test'],overwrite=True)
        else:
            best_metric=demjson3.decode_file('best_metric.json')
            if type == 'regression':
                if best_metric < metrics['test']:
                    demjson3.encode_to_file('best_metric.json',metrics['test'],overwrite=True)
                    best_model.save_model('best_model.json')
                    # with open(args.pipeline_path, 'rb') as f:
                    #     pipeline = pickle.load(f)
                    # with open('best_plan.pkl', 'wb') as f:
                    #     pickle.dump(pipeline, f)
            else:
                if metrics['test'] > best_metric:
                    demjson3.encode_to_file('best_metric.json',metrics['test'],overwrite=True)
                    best_model.save_model('best_model.json')
                    # with open(args.pipeline_path, 'rb') as f:
                    #     pipeline = pickle.load(f)
                    # with open('best_plan.pkl', 'wb') as f:
                    #     pickle.dump(pipeline, f)
    # Print the result as a JSON string
    print(metrics['test'])
    # os.environ['RUN_RESUlT']=str(metrics['test'])