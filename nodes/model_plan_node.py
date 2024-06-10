import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore
from nodes.base_node import BasePromptNode,model_plan_template
import re
import os 


class ModelPlanProviderNode(BasePromptNode):

    def __init__(self,):
        super(ModelPlanProviderNode,self).__init__()
        
    def run(self, task_prompt):

        model_config=self.model_config
        prompt_config={'task_prompt':task_prompt}
        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            if "```json" in result:
                result = re.findall(r'```json\n(.*?)\n```', result, re.DOTALL)
            result_json = json.loads(result[0].replace("'", '"'))
            return result_json
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else: 
            info_dict = self.retry_operation(_operation,model_config,prompt_config)
        self.print_and_cache(info_dict['model_plan'],info_dict)
        save_path = self.root_dir / "model_plan.json"
        with open(save_path, "w") as f:
            json.dump(info_dict['model_plan'], f)
        return {self.output_names[0]: info_dict['model_plan'], "_debug": "model_plan"}, "output_1"



