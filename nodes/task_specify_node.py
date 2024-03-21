import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore
from regex import P
from nodes.base_node import BasePromptNode,task_specify_template
import re
import os



class TaskSpecifyNode(BasePromptNode):

    def __init__(self,):
        super(TaskSpecifyNode,self).__init__()

    def _operation(self,model_config,prompt_config):
        result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
                # Define regex pattern to match JSON string
        pattern = r'```json\s*([\s\S]*?)```'

        # Search for pattern in input string
        match = re.search(pattern, result)

        # Extract JSON string from match object
        result = match.group(1).replace("'", '"')
        
        result_json = json.loads(result)
        return result_json

        
    def run(self, inputs: List[Dict]) -> Tuple[Dict, str]:

        input_dict =  self.parse_inputs(inputs)
        data_description = input_dict["data_description"]
        query = input_dict["query"]
        model_config=self.model_config
        prompt_config={"query": query, "data_description": data_description}
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            info_dict= self.retry_operation(self._operation,model_config,prompt_config)
        self.print_and_cache(info_dict['specific_task'],info_dict)

        return {self.output_names[0]: info_dict['specific_task'], "_debug": "specific task"}, 'output_1'

