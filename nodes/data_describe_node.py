import pandas as pd
from typing import List,Tuple
from nodes.base_node import BasePromptNode
from haystack.schema import Document
import demjson3
import re
import os 


class DataDescribeNode(BasePromptNode):

    def __init__(self):
        super(DataDescribeNode,self).__init__()

    def run(self, query: str, documents: List[Document]) -> Tuple[dict, str]:

        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            pattern = r'```json\s*([\s\S]*?)```'
            # Search for pattern in input string
            match = re.search(pattern, result)
            # Extract JSON string from match object
            result = match.group(1).replace("'", '"')
            print(result)
            target_column = demjson3.decode(result)['target_column']

            return {'target_column':target_column}

        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
        else:
            origin_data = documents[0].content
            file_name=documents[0].meta['name']
            data_samples=self.generate_data_description(origin_data,file_name)
            data_description = documents[1].content + "\n" +data_samples
            prompt_config={'data_samples':data_samples,"data_description":data_description}
            info_dict = self.retry_operation(_operation,self.model_config,prompt_config)
            info_dict['data_description'] = data_description
        target_column=info_dict['target_column']
        df=documents[0].content
        df = df.rename(columns={target_column: 'target'})
        
        df.to_pickle('data.pkl')
        self.print_and_cache(info_dict['data_description'],info_dict)
        return {self.output_names[0]: info_dict['data_description'],'documents':documents, "_debug": "data described"}, 'output_1'


    def generate_data_description(self, data: pd.DataFrame,file_name:str) -> str:
        # Your code to generate data description
        samples = ""
        df_ = data.head(5)
        for i in list(df_):
            # show the list of values
            nan_freq = "%s" % float("%.2g" % (data[i].isna().mean() * 100))
            s = df_[i].tolist()
            if str(data[i].dtype) == "float64":
                s = [round(sample, 2) for sample in s]
            samples += (
                f"{df_[i].name} ({data[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
            )
        return f"\nColumns in {file_name} (true feature dtypes listed here, categoricals encoded as int):{samples}"
