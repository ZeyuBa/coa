from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore
from nodes.base_node import BasePromptNode
import os 


class TaskPromptNode(BasePromptNode):

    def __init__(self):
        super(TaskPromptNode,self).__init__()

    def run(self, inputs: List[Dict]):
        input_dict =  self.parse_inputs(inputs)
        task_type = input_dict['type']["task_type"]
        specific_task = input_dict['specific_task']
        def _operation(model_config,prompt_config):
            task_prompt = specific_task + f"\nThis is a {task_type} task."
            return {'task_prompt':task_prompt}
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            info_dict = self.retry_operation(_operation,{},{})
        self.print_and_cache(info_dict['task_prompt'],info_dict)
        return {self.output_names[0]: info_dict['task_prompt'], "_debug": {"anything": "you want"}}, "output_1"

