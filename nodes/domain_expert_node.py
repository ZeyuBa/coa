from nodes.base_node import BasePromptNode
import re
import os 
import json_repair
import json

class DomainExpertNode(BasePromptNode):
    def __init__(self):
        super(DomainExpertNode,self).__init__()
        self.is_retry=True
    def run(self, inputs):
        input_dict = self.parse_inputs(inputs)
        data_description=input_dict["data_description"]["desc"]
        data_report=input_dict['type']["data_report"]
        model_config=self.model_config
        prompt_config={'data_description':data_description,'data_report':data_report,}
        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            if "```json" in result:
                pattern = r'```json\s*([\s\S]*?)```'
                match = re.search(pattern, result)
                result = match.group(1).replace("'", '"')
            try:
                result_json =json.loads(result)
                return result_json
            except:
                good_json_string = json_repair.repair_json(result, skip_json_loads=True)
                result_json =json.loads(good_json_string)
                if isinstance(result_json,dict):
                    return result_json
            raise ValueError("The result is not a valid json format.")
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            info_dict={}
            info_dict['domain_knowledge'] = self.retry_operation(_operation,model_config,prompt_config)
        self.print_and_cache(info_dict['domain_knowledge'],info_dict)
        return {self.output_names[0]: info_dict['domain_knowledge'], "_debug": "domain_knowledge"}, "output_1"

