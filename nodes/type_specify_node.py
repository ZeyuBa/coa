import json
import logging
from statistics import mode
from typing import Any, Dict, List, Optional, Tuple, Union
from nodes.base_node import BasePromptNode,task_type_template,TASK_TYPES
from haystack.schema import Document
from nodes.error_type_analysis import classification_report, regression_report
import re
import os 


class TypeSpecifyNode(BasePromptNode):
    def __init__(self, ):
        super(TypeSpecifyNode,self).__init__()
        self.support_task = TASK_TYPES.copy()


    def run(self, data_description, query:str,documents: List[Document]) -> Tuple[Dict, str]:
        data_description =data_description
        query = query
        model_config={"model_kwargs":{"temperature": 0,
                                    "max_retries":3,
                        "response_format": {"type": "json_object"}
                        }}
        prompt_config={"query": query, "data_description": data_description["desc"]}
        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            # result=self.response_ollama(model_config,prompt_config)
            if "```json" in result:
                
                result = re.findall(r'```json\n(.*?)\n```', result, re.DOTALL)

                result_json = json.loads(result[0].replace("'", '"'))
            else: result_json=json.loads(result)
            task_type = result_json['task_type'].lower()
            if task_type in TASK_TYPES:
                print(documents[0].content.iloc[:, :-1].columns)
                
                data_report = eval(f'{task_type}_report')(documents[0].content.iloc[:, :-1], documents[0].content.iloc[:, -1])
            
            return {"task_type":task_type,"data_report":data_report}
        
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            info_dict= self.retry_operation(_operation,model_config,prompt_config)
            
        self.print_and_cache(info_dict['task_type'],info_dict)

        return {self.output_names[0]: info_dict, "_debug": "task type specified"}, 'output_1'
