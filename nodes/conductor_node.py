from distutils.util import execute
import os
import pickle
import random
import subprocess
import time
import sys
from decimal import Decimal
from pathlib import Path
import inspect
import pandas as pd
from loguru import logger
from nodes.base_node import BasePromptNode
# from nodes.src.bo_space import AutoMLSpace

class ConductorNode(BasePromptNode):

    def __init__(self):
        super().__init__()
    
    def run(self, query='', inputs=None, documents=[]):

        
        data = pd.read_pickle(f'./{documents[3].content}/dev.pkl') if query == 'predict' else pd.read_pickle(f'./{documents[3].content}/train.pkl')
        features = data.columns if query == 'predict' else data.columns[:-1]
        # # Load the pipeline
        # with open(pipeline_path, 'rb') as f:
        #     pipeline = pickle.load(f)
        # # print(pipeline)
        # preprocessed_data = data
        # for op in pipeline:
        #     op[1]['data']=preprocessed_data
        #     if op[0]=='resampling':
        #         continue
        #     preprocessed_data=eval(op[0])(**op[1])
        # X = preprocessed_data.drop(columns='target')
        # y = preprocessed_data['target']
        
        if inputs:
            input_dict = self.parse_inputs(inputs)
            eda_plan=input_dict['eda_list']
            bo_plan=self.get_nested(input_dict,['bo_output','bo_plan'])
            func_name=self.get_nested(input_dict,['bo_output','func_name'])
            func_example=self.get_nested(input_dict,['bo_output','func_example'])
            space_list=self.get_nested(input_dict,['bo_output','space_list'])
            exec(f'from nodes.src.{func_name} import {func_name}')
            expected_args = inspect.getfullargspec(eval(func_name)).args
            try:
                execute('from nodes.src.bo_space import AutoMLSpace')
                sp=AutoMLSpace(space_list)
            except:
                sp=None
            sample=sp.sample_point()
            params_list=sp.decode_sample(sample,func_name,expected_args)
            for param in params_list:
                param['data']=data
                input_args={elem:param[elem] for elem in expected_args}
                data=eval(func_name)(**input_args)
            X,y=data.drop(columns=['target']),data['target']
            X_path,y_path=f'./{documents[3].content}/X.pkl',f'./{documents[3].content}/y.pkl'
            X.to_pickle(X_path)
            y.to_pickle(y_path)
            data.to_pickle(f'./{documents[3].content}/train.pkl')
            # conduct_plan = self._generate_conduct_plan(features, input_dict)
        else:
            pipeline = documents[1]
            conduct_plan = self._decode_pipeline(pipeline, features)
        model_path=f'./{documents[3].content}/info.json'
        info_dict = self._execute_pipeline(data_path=[X_path,y_path],query=query,model_path=model_path)
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

    def _decode_pipeline(self, data_path,pipeline, features):
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

    def _execute_pipeline(self,data_path, query,model_path):
        python_interpreter = sys.executable
        script_path = Path(__file__).parent.resolve() / 'src' / 'run.py'
        # pipeline_path = 'best_plan.pkl' if query == 'predict' else 'conduct_plan.pkl'

        # with open(pipeline_path, 'wb') as f:
        #     pickle.dump(conduct_plan, f)

        start = time.time()
        # data_path = document if isinstance(document, str) else 'train.pkl'
        X_path,y_path=data_path
        command = [python_interpreter, script_path, X_path,y_path, model_path, query]

        try:
            process_output = subprocess.run(command, capture_output=True, text=True)
            if process_output.returncode != 0:
                self.logger.info(f"ERROR: {process_output.stderr.replace('<module>', '')}")
        except subprocess.SubprocessError as e:
            print(e)

        self.logger.info(f'time cost: {time.time() - start}')
        return {'result': Decimal(process_output.stdout.strip())}
