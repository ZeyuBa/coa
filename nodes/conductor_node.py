import os
import pickle
import random
import subprocess
import time
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

from nodes.base_node import BasePromptNode

class ConductorNode(BasePromptNode):

    def __init__(self):
        super().__init__()
    
    def run(self, query='', inputs=None, documents=[]):
        data = self._load_data(documents[0])
        features = data.columns if query == 'predict' else data.columns[:-1]
        
        if inputs:
            input_dict = self.parse_inputs(inputs)
            conduct_plan = self._generate_conduct_plan(features, input_dict)
        else:
            pipeline = documents[1]
            conduct_plan = self._decode_pipeline(pipeline, features)

        info_dict = self._execute_pipeline(query, documents[0], conduct_plan)
        self.print_and_cache(info_dict['result'], info_dict)

        return {self.output_names[0]: info_dict['result'], "_debug": "code"}, 'output_1'

    def _load_data(self, document):
        if isinstance(document, str):
            return pd.read_pickle(document)
        return document.content

    def _generate_conduct_plan(self, features, input_dict):
        n_components = random.randint(1, len(features) - 1)
        conduct_plan = []
        for key, value in input_dict['data_plan'].items():
            if value['selected']:
                if key == 'feature_selection':
                    selected_feature = [random.choice(features)]
                    conduct_plan.append((key, {'selected_feature': selected_feature}))
                elif key in ['dimensionality_reduction', 'resampling']:
                    continue
                else:
                    conduct_plan.append((key, {'methods': value['option']}))
        return conduct_plan

    def _decode_pipeline(self, pipeline, features):
        conduct_plan = []
        for key, value in pipeline:
            if key == 'feature_selection':
                selected_feature = [features[i] for i in value]
                conduct_plan.append((key, {'selected_feature': selected_feature}))
            elif key == 'dimensionality_reduction':
                conduct_plan.append((key, {'methods': value[0], 'n_components': value[1]}))
            else:
                conduct_plan.append((key, {'methods': value}))
        return conduct_plan

    def _execute_pipeline(self, query, document, conduct_plan):
        python_interpreter = sys.executable
        script_path = Path(__file__).parent.resolve() / 'src' / 'run.py'
        pipeline_path = 'best_plan.pkl' if query == 'predict' else 'conduct_plan.pkl'

        with open(pipeline_path, 'wb') as f:
            pickle.dump(conduct_plan, f)

        start = time.time()
        data_path = document if isinstance(document, str) else 'data.pkl'
        command = [python_interpreter, script_path, data_path, 'info.json', pipeline_path, query]

        try:
            process_output = subprocess.run(command, capture_output=True, text=True)
            if process_output.returncode != 0:
                self.logger.info(f"ERROR: {process_output.stderr.replace('<module>', '')}")
        except subprocess.SubprocessError as e:
            print(e)

        self.logger.info(f'time cost: {time.time() - start}')
        return {'result': Decimal(process_output.stdout.strip())}
