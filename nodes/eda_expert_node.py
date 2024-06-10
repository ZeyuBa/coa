#do complete EDA before training

#0. remove undesired features in train and test
#1. duplicates operation: [drop, replace_with_median]
#2. features and label distribution visualization--> sparse features

#3. new feature construction [threshold_clip, bool_encoder]

# 3. partial dependency analysis:
# more trics for coordinates data


#4. more models like random forest other than linear model
#5. partial dependency analysis?
#6. BO on each model and model esemble





import demjson3
from typing import Any, Dict, List, Optional, Tuple, Union
from nodes.base_node import BasePromptNode
from nodes.base_node import eda_template2
import re
import os 
import json_repair
import json
class EDAExpertNode(BasePromptNode):

    def __init__(self):
        super(EDAExpertNode,self).__init__()
        self.is_retry=True
    def run(self, inputs):
        with open(self.root_dir/'nodes/code_template/feat_encoder.py','r',encoding='utf-8') as f:
            code_snippet=f.read()
        # Regex pattern to match the content between `def` and `pass`
        pattern = r'def[^\S\n]*\w+\([^)]*\)\s*->\s*[^:]*:\s*\"\"\"[^\"]*\"\"\"\s*pass'
        matches = re.findall(pattern, code_snippet, re.MULTILINE)
        tool_desc=matches[0]
        inputs_dict=self.parse_inputs(inputs)
        important_feature_info=inputs_dict['data_description']['other_info']
        domain_knowledge=inputs_dict['domain_knowledge']
        tool_desc=tool_desc

        model_config=self.model_config
        prompt_config={'domain_knowledge':domain_knowledge,'feature_info':important_feature_info,'tools':tool_desc}
        BO=True

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
                # result_json=json_repair.repair_json(result)
                if isinstance(result_json,dict):
                    return result_json
            # result_json=demjson3.decode(result)
            raise ValueError("The result is not a valid json format.")
        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            info_dict = self.retry_operation(_operation,model_config,prompt_config)
            feat_list = info_dict['feat_list']
            info_dict['eda_list']=[]
            self.prompt_template=eda_template2
            for feat in feat_list:
                prompt_config['feature_name']=feat.split('_')[0]
                info_dict['eda_list'].append(self.retry_operation(_operation,model_config,prompt_config))            
        self.print_and_cache(info_dict['eda_list'],info_dict)
        return {self.output_names[0]: info_dict['eda_list'], "_debug": "domain_knowledge"}, "output_1"

