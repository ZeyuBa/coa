import demjson3
from typing import Any, Dict, List, Optional, Tuple, Union
from nodes.base_node import BasePromptNode,domain_knowledge_template
import re
import os 

import json
class DomainExpertNode(BasePromptNode):

    def __init__(self):
        super(DomainExpertNode,self).__init__()
        self.is_retry=True
    def run(self, inputs):

        input_dict = self.parse_inputs(inputs)

        task_prompt=input_dict['task_prompt']
        data_description=input_dict["data_description"]
        data_report=input_dict['type']["data_report"]
        model_config=self.model_config
        prompt_config={'data_description':data_description,'data_report':data_report,'task_prompt':task_prompt}
        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
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
            info_dict = self.retry_operation(_operation,model_config,prompt_config)
        self.print_and_cache(info_dict['domain_knowledge'],info_dict)
        return {self.output_names[0]: info_dict['domain_knowledge'], "_debug": "domain_knowledge"}, "output_1"

