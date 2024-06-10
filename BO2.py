from enum import auto
import pandas as pd
from hebo.design_space.design_space import DesignSpace

class AutoMLSpace:
    def __init__(self, space_list):
        self.space_list = space_list
        self._create_search_space()

    def _create_search_space(self):
        """Parse the space list to design space"""
        self.space=DesignSpace().parse(self.space_list)


    def sample_point(self):
        """Return a random sample from the space."""
        return self.space.sample()

    def decode_sample(self, sample: pd.DataFrame):
        """
        Decode the given sample from the BO iteration into a list of parameter dictionaries
        suitable for applying the 'feat_encoder' function.
        
        :param sample: DataFrame with sampled parameters.
        :return: List of parameter dictionaries.
        """
        param_dicts = []
        operation_settings = [
            {'feature_name': 'AgeInDays', 'threshold_cols': 'AgeInDays_threshold', 'method_cols': 'AgeInDays_method'},
            {'feature_name': 'SuperplasticizerComponent', 'threshold_cols': 'SuperplasticizerComponent_threshold', 'method_cols': 'SuperplasticizerComponent_method'},
            {'feature_name': 'FlyAshComponent', 'threshold_cols': 'FlyAshComponent_threshold', 'method_cols': 'FlyAshComponent_method'}
        ]
        
        for setting in operation_settings:
            param_dict = {
                'feature_name': setting['feature_name'],
                'threshold': sample[setting['threshold_cols']].values[0] if setting['threshold_cols'] in sample else '',
                'method': sample[setting['method_cols']].values[0] if setting['method_cols'] in sample else ''
            }
            param_dicts.append(param_dict)
        
        return param_dicts

# Example of space list (this would be defined outside and passed into AutoMLSpace)
space_list = [
    {'name': 'AgeInDays_threshold', 'type': 'cat', 'categories': [30, 60]},
    {'name': 'AgeInDays_method', 'type': 'cat', 'categories': ['clip', 'bool']},
    {'name': 'SuperplasticizerComponent_threshold', 'type': 'cat', 'categories': [1.5, 2.0]},
    {'name': 'SuperplasticizerComponent_method', 'type': 'cat', 'categories': ['clip', 'bool']},
    {'name': 'FlyAshComponent_threshold', 'type': 'cat', 'categories': [0.5, 1.5]},
    {'name': 'FlyAshComponent_method', 'type': 'cat', 'categories': ['clip', 'bool']}
]

# Create an instance of AutoMLSpace
automl_space = AutoMLSpace(space_list)
print(automl_space.space_list)
print(automl_space.sample_point())

