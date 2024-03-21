from nodes.datalab.datalab import Datalab
import pandas as pd
# Create a DataFrame with the features
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from typing import Union, List, Any
num_crossval_folds = 5


def classification_report(data: pd.DataFrame, label: Any):
    clf = HistGradientBoostingClassifier()

    pred_probs = cross_val_predict(
        clf,
        data,
        label,
        cv=num_crossval_folds,
        method="predict_proba",
    )
    data_df = pd.DataFrame(data)
    data_df['label'] = label
    KNN = NearestNeighbors(metric='euclidean')
    KNN.fit(data)
    knn_graph = KNN.kneighbors_graph(mode="distance")

    lab = Datalab(data_df, label_name='label', verbosity=2)

    from colorama import Fore

    lab.find_issues(pred_probs=pred_probs, knn_graph=knn_graph)

    return lab.report(verbosity=2)


def regression_report(data: pd.DataFrame, label: Any):
    model = HistGradientBoostingRegressor()
    pred_probs = cross_val_predict(
        estimator=model,
        X=data,
        y=label,
        cv=num_crossval_folds,
    )
    data_df = pd.DataFrame(data)
    data_df['label'] = label

    # knn_graph = KNN.kneighbors_graph(mode="distance")
    import warnings

    # Filter out specific warnings temporarily
    warnings.simplefilter("ignore", category=RuntimeWarning)
    lab = Datalab(data_df, label_name='label')
    lab.find_issues(pred_probs=pred_probs)
    warnings.resetwarnings()
    return lab.report()


if '__main__' == __name__:
    from sklearn.datasets import load_iris, load_diabetes
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    report = classification_report(iris_df.iloc[:, :-1], iris_df.iloc[:, -1])
    print(type(report))
    # diab=load_diabetes()
    # diab_df=pd.DataFrame(diab.data,columns=diab.feature_names)
    # diab_df['target']=diab.target
    # print(diab_df.iloc[:,:-1])
    # report=regression_report(diab_df.iloc[:,:-1],diab_df.iloc[:,-1])
