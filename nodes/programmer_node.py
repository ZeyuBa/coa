from nodes.base_node import BasePromptNode,program_template,refine_template
import re
from pathlib import Path
root_dir=Path(__file__).parent.resolve()
code_template_dir=root_dir / "code_template"
save_path=root_dir/ "src"
import os
from loguru import logger
class ProgrammerNode(BasePromptNode):

    def __init__(self, ):
        super(ProgrammerNode,self).__init__()
        self.is_cache=True

    def _operation(self, model_config, prompt_config):
        result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
        pattern = r"```python(.*?)```"
        maches = re.findall(pattern, result, re.DOTALL)
        return '\n'.join(maches)

    def run(self, inputs):
        input_dict = self.parse_inputs(inputs)
        task_prompt=input_dict['task_prompt']
        data_plan=[(k,v['detail']) for k,v, in input_dict['data_plan'].items()] 
        bo_plan=self.get_nested(input_dict,['bo_output','bo_plan'])
        logger.info('bo_plan: ',bo_plan)
        func_name=self.get_nested(input_dict,['bo_output','func_name'])
        func_example=self.get_nested(input_dict,['bo_output','func_example'])
        model_config=self.model_config
        code_plan={'data_plan':data_plan,}
        if bo_plan:
            code_plan.update({'bo_plan':[('bo_space',bo_plan)]})
        logger.info(code_plan)
        for type,op_l in code_plan.items():
            for op in op_l:
                file_name=op[0]+".py"
                if os.path.exists(code_template_dir / file_name):
                    with open(code_template_dir / file_name, mode="r", encoding="utf-8") as f:
                        temp=f.read()
                if type == 'data_plan':
                    prompt_config={'code_task':'','code_template':temp,
                                #    "detail":op[1]
                                # "task_prompt":task_prompt
                                }
                elif type == 'bo_plan':
                    import inspect
                    exec(f'from nodes.src.{func_name} import {func_name}')

                    expected_args = inspect.getfullargspec(eval(func_name)).args
                    new_line = f"\n        Note: func_name: {func_name}. Expected arguments are: {expected_args}. {func_example}"
                    logger.info(new_line)
                    # Regular expression to find and replace the docstring
                    updated_temp = re.sub(
                        r'(def decode_sample.*?\"\"\".*?)(\"\"\")', 
                        r'\1' + new_line + r'\2', 
                        temp, 
                        flags=re.DOTALL
                    )

                    logger.info(updated_temp)

                    prompt_config={'code_task':op[1],'code_template':updated_temp,
                                #    "detail":op[1]
                                # "task_prompt":task_prompt
                                }

                if os.path.exists(save_path / file_name) and self.is_cache:
                    with open(save_path / file_name, mode="r", encoding="utf-8") as f:
                        script=f.read()
                    self.print_and_cache(f'{file_name} already exists',{})
                    continue


                result= self.retry_operation(self._operation,model_config,prompt_config)
                self.prompt_template=refine_template
                for _ in range(self.max_retries):   
                    with open(save_path / file_name, mode="w", encoding="utf-8") as f:
                        f.write(result)
                    result,error=execute_code(file_name)
                    print(result,error)
                    if error is None: break 
                    else: 
                        prompt_config={'code_block':result,
                                    #    "detail":str(op),
                                    "error_message":error,
                                    "code_template":updated_temp if updated_temp else temp
                                        }
                        self.model_config['model_kwargs']['temperature']=0.3
                        result=self.retry_operation(self._operation,self.model_config,prompt_config)
                self.prompt_template=program_template
                if result:
                    # functions+=file_name+'\n\n'+result.split('if __name__ == "__main__":')[0]+'\n'
                    self.print_and_cache(f'{file_name} saved',{})
                else:
                    self.logger.error(f'Error in code generation!--{error}')

        return {self.output_names[0]:save_path / file_name, "_debug": "code"}, 'output_1'

import subprocess
import sys
def execute_code(file_name):
    # Path to the Python interpreter (the one currently executing the script)
    python_interpreter = sys.executable

    # Path to the Python script you want to execute
    script_path = save_path / file_name
    with open(save_path / file_name, mode="r", encoding="utf-8") as f:
        script=f.read()
    # Executing the script
    try:
        result = subprocess.run([python_interpreter, script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return script,None
    except subprocess.CalledProcessError as e:
        return script,e.stderr

