import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from colorama import Fore
from nodes.base_node import BasePromptNode,agent_generation_template
import re

class AgentGeneratorNode(BasePromptNode):

    def __init__(self, verbose=True, outputs: List[str] = ['output_1']):
        super().__init__(agent_generation_template, verbose, outputs)
        
    def run(self, task_prompt):

        model_config={"model_kwargs":{"temperature":1,"max_tokens": 1000,
                          "response_format": {"type": "json_object"}}}
        prompt_config={'task_prompt':task_prompt}
        result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
        print("#############",result)

        result = re.findall(r'```json\n(.*?)\n```', result, re.DOTALL)
        print("#############",result)
        try:
            result_json = json.loads(result[0].replace("'", '"'))

            if self.verbose:
                print(Fore.BLACK +
                      f"{self.__class__.__name__} message:\n{result_json['agent_list']}\n")

            return {self.output_names[0]: result_json['agent_list'], "_debug": "agent_list"}, 'output_1'
        except json.JSONDecodeError:
            logging.error("JSON format error in TaskSpecifierNode")
            return {}, 'output_1'


