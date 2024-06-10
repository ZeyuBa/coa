

import demjson3
from typing import Any, Dict, List, Optional, Tuple, Union
from nodes.base_node import BasePromptNode,bo_code_template
from nodes.code_template.bo_space import AutoMLSpace

import re
import os 
import json
from loguru import logger
use_example='''
from BO import AutoMLSpace
sp=AutoMLSpace(space_list)
from hebo.optimizers.hebo import HEBO
import numpy as np
def objective(params : pd.DataFrame) -> np.ndarray:
    result=BOpipeline.run(query='train',documents=['train.pkl',sp.decode_sample(params)])
    return -np.array([[result['result']]])
'''
class BONode(BasePromptNode):

    def __init__(self):
        super(BONode,self).__init__()
        self.is_retry=True
        self.is_cache=True
    def run(self, eda_list):
        # logger.info('TEST')
        with open(self.root_dir/'nodes/src/feat_encoder.py','r',encoding='utf-8') as f:
            code_snippet=f.read()
        tool_desc=code_snippet.split('if __name__ == "__main__":')[0]
        def _operation(model_config,prompt_config):
            # logger.info(self.model_type)
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            if "```json" in result:
                pattern = r'```json\s*([\s\S]*?)```'
                match = re.search(pattern, result)
                result = match.group(1).replace("'", '"')
            try:
                result_json = json.loads(result)
                return result_json
            except:
                result_json = demjson3.decode(result)
                if isinstance(result_json,dict):
                    return result_json
            # result_json=demjson3.decode(result)
            raise ValueError("The result is not a valid json format.")
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            res_list=[]
            model_config=self.model_config
            for op in eda_list:
                prompt_config={'operation_list':op['params_dict'],"tools":tool_desc}
                tmp=self.retry_operation(_operation,model_config,prompt_config)['hyperparam_dict']
                logger.info(tmp)
                res_list+=tmp
            info_dict={'space_list':res_list}

        self.print_and_cache(f"Space list:{info_dict['space_list']}",info_dict)
        #prefill the prompt and return to programm node

        func_name = re.findall(r"\bdef (\w+)", tool_desc)[0]
        exec(f'from nodes.src.{func_name} import {func_name}')
        func_docstring=eval(func_name).__doc__
        self.prompt_template=bo_code_template
        prompt_config={"func_name":func_name,"func_desc":func_docstring,"use_example":use_example,"space_list":info_dict['space_list']}
        bo_code_task=self.prompt_template.format(**prompt_config)
        op_exmaple=eda_list[0]['params_dict']
        logger.info(op_exmaple)
        tmp_dict={}
        for k,v in op_exmaple.items():
            tmp_dict[k]=v[0] if isinstance(v,list) else v
        func_example=f'''
            Input args example:{tmp_dict}
        '''
        

        return {self.output_names[0]: {'bo_plan':bo_code_task,'func_name':func_name,'func_example':func_example,'space_list':info_dict['space_list']}, "_debug": "domain_knowledge"}, "output_1"

