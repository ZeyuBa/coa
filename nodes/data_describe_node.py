from calendar import c
import pandas as pd
from typing import List,Tuple
from nodes.base_node import BasePromptNode
from haystack.schema import Document
import demjson3
import re
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from loguru import logger
import json
from pathlib import Path


class DataDescribeNode(BasePromptNode):

    def __init__(self):
        super(DataDescribeNode,self).__init__()

    def run(self, query: str, documents: List[Document]) -> Tuple[dict, str]:

        def _operation(model_config,prompt_config):
            result = self.initialize_prompt_node(model_config,prompt_config=prompt_config)
            # result=self.response_ollama(model_config,prompt_config)
            if '```json' in result:
                pattern = r'```json\s*([\s\S]*?)```'
                # Search for pattern in input string
                match = re.search(pattern, result)
                # Extract JSON string from match object
                result = match.group(1).replace("'", '"')
            try: 
                target_column = json.loads(result)['target_column']
            except Exception as e:
                logger.error(e)
                target_column = demjson3.decode(result)['target_column']
            return {'target_column':target_column}
        hist=''
        data_description=''

        if os.path.exists(self.cache_file) and self.is_cache:
            info_dict=self.load_cache()
            logger.info(info_dict)
        else:
            data_description = documents[1].content 
            prompt_config={"data_description":data_description}
            info_dict = self.retry_operation(_operation,self.model_config,prompt_config)
        target_column=info_dict['target_column']
        train_df= documents[0].content
        test_df= documents[2].content
        train_df = train_df.rename(columns={target_column: 'target'})
        file_name=documents[0].meta.get('name','')
        data_samples_desc,data_info=self.generate_data_description(train_df,test_df,file_name)        
        info_dict['data_description'] = data_description

        train_df.to_pickle(f'./{documents[3].content}/train.pkl')

        self.draw_hist_diagram(train_df,test_df,target_column,documents)
        self.print_and_cache(info_dict['data_description'],info_dict)
        output_dict={'desc':info_dict['data_description']+'\n'+data_samples_desc,"other_info":data_info}
        return {self.output_names[0]: output_dict, "_debug": "data described"}, 'output_1'


    def generate_data_description(self, train: pd.DataFrame,test:pd.DataFrame,file_name:str) -> str:
        # Your code to generate data description
        samples = ""
        data=train
        df_ = data.head(5)
        data_info={}
        for i in list(df_):
            # show the list of values
            nan_freq = "%s" % float("%.2g" % (data[i].isna().mean() * 100))
            
            s = df_[i].tolist()
            if str(data[i].dtype) == "float64":
                s = [round(sample, 2) for sample in s]
            samples += (
                f"{df_[i].name} ({data[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
            )
        # samples+=data.info()
        # print(data.columns[:-1])
        data_info['duplicate']=''
        duplicate = data[data.columns[1:-1]].duplicated().sum()    
        data_info['duplicate']+=(
                f"duplicate-num in train: {duplicate:4}\n"
            )
        data_info['duplicate']+=(
                f"duplicate-num in test: {test.duplicated().sum():4}\n"
            )
        
                    # Assuming 'train' is your DataFrame
        bins=20
        for col in data.columns:
            # Use pd.cut to bin the data into 20 equal-sized bins
            # The 'right=False' parameter includes the left edge of each bin and excludes the right edge
            binned_data = pd.cut(data[col], bins=bins, right=False)
            
            # Count the occurrences in each bin
            bin_counts = binned_data.value_counts().sort_index()
            
            # Print the column name for clarity
            hist=(f"--- {col} Distribution in {bins} Intervals ---")
            # Print the counts for each bin
            hist+=str(bin_counts)
            hist+=("\n")  # Add a newline for better readability between columns
        data_info['hist']=hist
        data_info['skewness']='--- Skewness for all features ---\n'
        data_info['skewness']+=train.skew().sort_values(key=lambda x: abs(x),ascending=False).to_string()
        data_info['kurtosis']='--- Kurtosis for all features ---\n'
        data_info['kurtosis']=train.kurt().sort_values(ascending=False).to_string()
        return f"\nColumns in {file_name} (true feature dtypes listed here):{samples}",data_info

    def analyze(self,train, test, col, ax,target):
        """Plot a histogram for column col into axes ax"""
        bins = 40
        column = train[col]
        if col in test.columns:
            both = np.hstack([column.values, test[col].values])
        else:
            both = column
        uni = np.unique(column)
        unival = len(uni)
        if unival < bins:
            vc_tr = column.value_counts().sort_index() / len(train)
            if col in test.columns:
                vc_te = test[col].value_counts().sort_index() / len(test)
                ax.bar(vc_tr.index, vc_tr, width=6, label='train', alpha=0.5)
                ax.bar(vc_te.index, vc_te, width=6, label='test', alpha=0.5)
            else:
                ax.bar(vc_tr.index, vc_tr, label='train', alpha=0.5)
            if unival <= 12:
                ax.set_xticks(vc_tr.index)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
            ax.set_xlabel(col + (' (target)' if col == target else ''))
            ax.set_ylabel('density')
            ax.legend()
        else:
            hist_bins = np.linspace(both.min(), both.max(), bins+1)
            ax.hist(column, bins=hist_bins, density=True, label='train', alpha=0.5)
            if col in test.columns:
                ax.hist(test[col], bins=hist_bins, density=True, label='test', alpha=0.5)
            ax.set_xlabel(col + (' (target)' if col == target else ''))
            ax.set_ylabel('density')
            ax.legend()
    def draw_hist_diagram(self,tran_df,test_df,target,documents):
        target = target
        _, axs = plt.subplots(3, 3, figsize=(12, 10))
        axs = axs.ravel()
        for col, ax in zip(tran_df.columns, axs):
            self.analyze(tran_df, test_df, col, ax,target)
        plt.tight_layout(h_pad=0.5, w_pad=0.5)
        plt.savefig(f'./{documents[3].content}/feat_dist.png')