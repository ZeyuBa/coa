
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore
from sklearn import pipeline
from sklearn.decomposition import PCA
from nodes.base_node import BasePromptNode,program_template,refine_template
from nodes.prompt_template import operation
from haystack.schema import Document
import re
from pathlib import Path
root_dir=Path(__file__).parent.resolve()
code_template_dir=root_dir / "code_template"
save_path=root_dir/ "src"
import os
class ProgrammerNode(BasePromptNode):

    def __init__(self, ):
        super(ProgrammerNode,self).__init__()
        self.is_cache=True

    def _operation(self, model_config, prompt_config):
        result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
        pattern = r"```python(.*?)```"
        maches = re.findall(pattern, result, re.DOTALL)

        return '\n'.join(maches)

    def run(self, inputs):
        input_dict = self.parse_inputs(inputs)
        task_prompt=input_dict['task_prompt']
        data_plan=[(k,v['detail']) for k,v, in input_dict['data_plan'].items()] 
        model_config=self.model_config
        code_plan=data_plan
        functions=''
        

        for op in code_plan:
            file_name=op[0]+".py"
            if os.path.exists(code_template_dir / file_name):
                with open(code_template_dir / file_name, mode="r", encoding="utf-8") as f:
                    temp=f.read()

            if os.path.exists(save_path / file_name) and self.is_cache:
                with open(save_path / file_name, mode="r", encoding="utf-8") as f:
                    script=f.read()
                self.print_and_cache(f'{file_name} already exists',{})
                functions+=file_name+'\n\n\n'+script.split('if __name__ == "__main__":')[0]+'\n\n'
                continue
            # if "run" in op[0]:
            #     self.prompt_template=main_func_template
            #     prompt_config={"code_template":temp,"detail":functions}
            # else: 
            prompt_config={'code_template':temp,
                        #    "detail":op[1]
                        # "task_prompt":task_prompt
                           }
            result= self.retry_operation(self._operation,model_config,prompt_config)
            self.prompt_template=refine_template
            for _ in range(self.max_retries):   
                with open(save_path / file_name, mode="w", encoding="utf-8") as f:
                    f.write(result)
                result,error=execute_code(file_name)
                print(result,error)
                if error is None: break 
                else: 
                    prompt_config={'code_block':result,
                                #    "detail":str(op),
                                   "error_message":error,
                                   "code_template":temp
                                    }
                    self.model_config['model_kwargs']['temperature']=0.3
                    result=self.retry_operation(self._operation,self.model_config,prompt_config)
            self.prompt_template=program_template
            if result:
                functions+=file_name+'\n\n'+result.split('if __name__ == "__main__":')[0]+'\n'
                self.print_and_cache(f'{file_name} saved',{})
            else:
                self.logger.error(f'Error in code generation!--{error}')

        return {self.output_names[0]:save_path / file_name, "_debug": "code"}, 'output_1'

import subprocess
import sys
def execute_code(file_name):
    # Path to the Python interpreter (the one currently executing the script)
    python_interpreter = sys.executable

    # Path to the Python script you want to execute
    script_path = save_path / file_name
    with open(save_path / file_name, mode="r", encoding="utf-8") as f:
        script=f.read()
    # Executing the script
    try:
        result = subprocess.run([python_interpreter, script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return script,None
    except subprocess.CalledProcessError as e:
        return script,e.stderr

