from pydoc import doc
from nodes.base_node import BasePromptNode
import subprocess
import pickle
from pathlib import Path
import random 
import os
root_dir=Path(__file__).parent.resolve()
print(root_dir)
code_template_dir=root_dir / "code_template"
save_path=root_dir/ "src"
import pandas as pd
import sys
python_interpreter = sys.executable
script_path = save_path / 'run.py'
from decimal import Decimal
import time
class ConductorNode(BasePromptNode):

    def __init__(self):
        super(ConductorNode,self).__init__()
        
    def run(self, query='',inputs=None, documents=[]):
        """
        write the overall code to include all preprocess and train evaluate pipeline.
        """
        if isinstance(documents[0], str):
            data=pd.read_pickle(documents[0])
        else:
            data=documents[0].content
        if query == 'predict':
            features=data.columns
            self.logger.info(f'features:{features}')
        else:
            features=data.columns[:-1]
        n_components=random.randint(1,len(features)-1)
        if inputs:
            input_dict=self.parse_inputs(inputs)
            self.data_plan=input_dict['data_plan']
            model_plan=input_dict['model_plan']
            conduct_plan=self.test(features,n_components)
            
        else: 
            pipeline=documents[1]
            # print(pipeline)
            conduct_plan=self.decode(pipeline,features)

        # print(conduct_plan)

        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            if query == 'predict':
                pipeline_path=  'best_plan.pkl'
            else : pipeline_path=  'conduct_plan.pkl'
            with open('conduct_plan.pkl', 'wb') as f:
                pickle.dump(conduct_plan, f)
            start=time.time()
            if isinstance(documents[0], str):
                command = [python_interpreter, script_path, documents[0],'info.json',pipeline_path,query]
            else:
                command = [python_interpreter, script_path, 'data.pkl','info.json',pipeline_path,query]
            try:
                # Wait for the process to complete, with a timeout
                _output=subprocess.run(command, capture_output=True, 
                                       text=True
                                       )
                if _output.returncode!=0:
                    # print(f"ERROR: {_output.stderr}")
                    self.logger.info(f"ERROR: {_output.stderr.replace('<module>','')}")
            except subprocess.SubprocessError as e:
                print(e)
            # result = subprocess.run([python_interpreter, script_path, 'data.pkl', 'info.json','conduct_plan.pkl'], capture_output=True, text=True)
            self.logger.info(f'time cost:{time.time()-start}')
            info_dict={'result':Decimal(_output.stdout.replace('\n',""))}
        # print(result.stdout)
        self.print_and_cache(info_dict['result'],info_dict)
        return {self.output_names[0]:info_dict['result'], "_debug": "code"}, 'output_1'
    
    def test(self,features,n_components):
        conduct_plan=[]
        for k,v in self.data_plan.items():
            if k=='feature_selection' and v['selected']:
                selected_feature=[features[random.randint(0,len(features))-1]]
                conduct_plan.append((k,{'selected_feature':selected_feature}))
            elif k=='dimensionality_reduction'and v['selected']:
                continue
                # conduct_plan.append((k,{'methods':'PCA','n_components':n_components}))
            elif k=='resampling':
                continue
                # conduct_plan.append((k,{'methods':'PCA','n_components':n_components}))
            elif v['selected'] :
                conduct_plan.append((k,{'methods':v['option']}))
        return conduct_plan
    def decode(self,pipeline,features):
        conduct_plan=[]
        for k,v in pipeline:
            if k=='feature_selection' :
                selected_feature=[features[i] for i in v]
                conduct_plan.append((k,{'selected_feature':selected_feature}))
            elif k=='dimensionality_reduction':
                conduct_plan.append((k,{'methods':v[0],'n_components':v[1]}))
            else: conduct_plan.append((k,{'methods':v}))

        return conduct_plan






