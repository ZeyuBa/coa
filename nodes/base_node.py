import os
from tabnanny import verbose
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from haystack.nodes.base import BaseComponent
from haystack.nodes import  PromptNode,PromptTemplate,AnswerParser
from jinja2 import Environment, FileSystemLoader
import itertools
from colorama import Fore, Style, init
import json
from loguru import logger
import yaml
from ollama import Client
import os
from openai import OpenAI
from abc import ABC, abstractmethod
from functools import reduce


TASK_TYPES = ["classification", "regression"]

_ = load_dotenv(find_dotenv())
default_model = "gpt-3.5-turbo"


# Initialize colorama
init(autoreset=True)

root_dir=Path(__file__).parent.parent.resolve()
import sys
sys.path.append(str(root_dir))
from nodes.prompt_template import MyPrompts

dir_path = Path(__file__).parent.resolve()
TEMPLATEDIR = dir_path

cache_path = dir_path/'.cache'
# Render specific blocks


pt=MyPrompts()
data_description_template=pt.get_prompt('data_description_template')
task_type_template = pt.get_prompt('task_type_template')
task_specify_template = pt.get_prompt('task_specify_template')
domain_knowledge_template = pt.get_prompt('domain_knowledge_template')
eda_template=pt.get_prompt('eda_template')
eda_template2=pt.get_prompt('eda_template2')
search_space_template=pt.get_prompt('search_space_template')
bo_code_template=pt.get_prompt('bo_code_template')
agent_generation_template = pt.get_prompt('agent_generation_template')
model_plan_template = pt.get_prompt('model_plan_template')
data_plan_template = pt.get_prompt('data_plan_template')
program_template = pt.get_prompt('program_template')
refine_template = pt.get_prompt('refine_template')

def load_config(config_file: str) -> dict:
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

config_path = Path(__file__).parent.resolve() / "config.yaml"
config = load_config(config_path)


class BasePromptNode(BaseComponent):
    outgoing_edges = 1
    colors = itertools.cycle([Fore.BLUE, Fore.GREEN, Fore.YELLOW, Fore.CYAN, Fore.MAGENTA,Fore.LIGHTBLUE_EX, Fore.LIGHTGREEN_EX, Fore.LIGHTYELLOW_EX, Fore.LIGHTCYAN_EX, Fore.LIGHTMAGENTA_EX])
    def __init__(self, config: dict=config):
        # common configs
        self.config=config.get('BasePromptNode')
        self.verbose = self.config.get('verbose', True)
        self.root_dir=root_dir



        # node specific configs
        self.config=config.get(self.__class__.__name__)
        self.model_type=self.config.get('model_type','gpt')
        self.max_retries = self.config.get('max_retries', 3)
        self.is_retry = self.config.get('is_retry', False)
        self.model_config=self.config.get('model_config',{})
        self.prompt_template = eval(self.config.get('prompt_template', '')) if 'prompt_template' in self.config else ''
        self.output_names = self.config.get('outputs')
        self.color = next(BasePromptNode.colors)  # Assign the next color in the cycle
        self.is_cache = self.config.get('is_cache',True)
        self.cache_file = cache_path / f"cache_{self.__class__.__name__}.json"
        self.logger = logger.opt(colors=True).bind(name=self.__class__.__name__)
        print(f"{self.__class__.__name__} initialized.")

    def parse_inputs(self, inputs: List[Dict]) -> Dict:
        return {k: v for i in inputs for k, v in i.items() if k != "_debug"}
    
    def initialize_prompt_node(self, model_config, prompt_config):
        if 'gpt' in self.model_type:
            self.clinet=None
            return self.gpt_node(model_config,prompt_config)
        elif 'ollama' in self.model_type:
            self.client =Client(host='http://localhost:11434')         
            return  self.ollama_node(model_config,prompt_config)
        elif 'openai' in self.model_type:
            self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
            return  self.openai_node(model_config,prompt_config)

    def gpt_node(self,model_config, prompt_config):
        node = PromptNode(model_name_or_path=model_config.get("model_name_or_path", default_model),
                          timeout=300,
                          api_key=os.getenv("OPENAI_API_KEY"),
                          model_kwargs=model_config.get("model_kwargs", {"temperature": 0}))
        prompt_template=PromptTemplate(self.prompt_template)
        return node.prompt(prompt_template=prompt_template, 
                           **prompt_config)[0]
    def ollama_node(self,model_config,prompt_config):
        format = 'json' if "response_format" in model_config['model_kwargs'] else ''
        response = self.client.chat(model='llama3:70b-instruct-q2_K',format=format, messages=[
        {
            'role': 'user',
            'content': self.prompt_template.format(**prompt_config),
        },
        ])

        return response['message']['content']

    def openai_node(self,model_config,prompt_config):
        if model_config['model_name_or_path']=='gpt-4':
            model="gpt-4-1106-preview"
        else: model=model_config['model_name_or_path']
        model_kwargs=model_config['model_kwargs']
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": self.prompt_template.format(**prompt_config),
                }
            ],
            model=model,**model_kwargs
        )
        print(chat_completion)
        return chat_completion.choices[0].message.content
    
    def load_cache(self):
        self.print_colored(f"Loading cache from {self.cache_file}")
        with open(self.cache_file, 'r',encoding='utf-8') as file:
            return json.load(file)
        
    def retry_operation(self, operation, *args, **kwargs):
            
        if not self.is_retry:
            return operation(*args, **kwargs)

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed in {self.__class__.__name__}: {e}")
                if self.model_type=='gpt':
                    self.model_config['model_kwargs']['do_sample']=True
                    self.model_config['model_kwargs']['temperature']=0.7
                    self.model_config['model_kwargs']['top_k']=20
                continue
        return {}, 'output_1'

    def cache_message(self, **kwarg):
        with open(self.cache_file, 'w',encoding='utf-8') as file:
            json.dump(kwarg, file)

    def print_colored(self, message):
        self.logger.info(self.color + "{} message:\n{}" + Style.RESET_ALL, self.__class__.__name__, message)
        # print(self.color+f"{self.__class__.__name__} message:\n{message}\n"+Style.RESET_ALL)

    def print_and_cache(self, message,return_dict):
        if self.verbose:
            self.print_colored(message)
        if not os.path.exists(self.cache_file) and self.is_cache:
            self.cache_message(**return_dict)
     

    def get_nested(self,dictionary, keys, default=''):
        # Use reduce to traverse the dictionary using the keys
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys, dictionary)

    

    def run_batch(self, queries: List[str], my_arg: Optional[int] = 10):
        ...
        output = {
            "documents": ...,
        }
        return output, "output_1"
    
if __name__=='__main__':
    node=BasePromptNode()
    node.prompt_template="{message}"
    model_config={"model_name_or_path":default_model,"model_kwargs":{"temperature": 0}}
    prompt_config={'message':"hello"}
    print(node.openai_node(model_config,prompt_config))