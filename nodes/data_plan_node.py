import os
from nodes.base_node import BasePromptNode
from nodes.prompt_template import operation
import re
from collections import OrderedDict
import demjson3
import json
class DataPlanProviderNode(BasePromptNode):

    def __init__(self, ):
        super(DataPlanProviderNode,self).__init__()
    def run(self, inputs,feedback=''):

        input_dict = self.parse_inputs(inputs)

        task_prompt=input_dict['task_prompt']
        data_description=input_dict["data_description"]["desc"]
        domain_knowledge=input_dict['domain_knowledge']

        operations=operation.__doc__.split('---')
        model_config=self.model_config
        data_plan={}
        def _operation(model_config,prompt_config):
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
            data_plan=self.load_cache()
        else:
            for op in operations:
                prompt_config={'operation':op,'data_description':data_description,'domain_knowledge':domain_knowledge,'task_prompt':task_prompt,'feedback':feedback}
                info_dict = self.retry_operation(_operation,model_config,prompt_config)
                for k,v in info_dict.items():
                    v.update({'detail':op})
                data_plan.update({"_".join(k.lower().split()):info_dict[k]})
        self.print_and_cache(data_plan,data_plan)
        return {self.output_names[0]: data_plan, "_debug": "data_plan"}, "output_1"



