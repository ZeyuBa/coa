# automl_pipeline.py

from math import inf
from pkg_resources import yield_lines
import yaml
from haystack import Pipeline
import argparse

from nodes import *  # Import all nodes
import pandas as pd
from haystack.schema import Document

from pathlib import Path

root_dir=Path(__file__).resolve().parents[0]
print(root_dir)

def is_terminated(old_metric,metric):
    pass 

def load_pipeline_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def initialize_node(node_type, **kwargs):
    if node_type in globals():
        return globals()[node_type]()
    # else:
    #     raise ValueError(f"Unknown node type: {node_type}")

def main():
    default_query="I want to perform the appropriate analysis with this dataset"
    parser = argparse.ArgumentParser(description='Process some input.')

    parser.add_argument('--data_type', type=str, default='kaggle', help='Specify the data type')
    parser.add_argument('--dir_path', type=str, default='playground-series-s3e9', help='Specify the directory path')
    parser.add_argument('--algo', type=str, default='BO', help='Specify the algorithm')
    parser.add_argument('--query', type=str, default=default_query, help='User query')
    parser.add_argument('--config_path', type=str, default='./pipeline_all.yaml', help='Specify the config file path')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Load cache or not')
    args = parser.parse_args()

    data_type=args.data_type
    dir_path=args.dir_path
    algo=args.algo
    query=args.query
    config_path=args.config_path
    no_cache=args.no_cache
    documents=[]
    pipeline_config=[]

    print(f'Data type: {args.data_type}')
    print(f'Directory path: {args.dir_path}')
    print(f'Algorithm: {args.algo}')
    print(f'Query: {args.query}')
    print(f'Config file path: {args.config_path}')

    if no_cache:
        cache_dir=root_dir / 'nodes' / '.cache'
        for file in cache_dir.glob('*'):
            # print(file)
            if file.is_file() or file.is_symlink():
                file.unlink()
            elif file.is_dir():
                file.rmdir()
    description=None
    ab_dir_path=root_dir / data_type/ 'tabular' / dir_path
    for file in ab_dir_path.glob('*'):
        if file.name=='train.csv':
            data=pd.read_csv(file)
            documents.append(Document(data, content_type='table',meta={'name':file.name}))
    for file in ab_dir_path.glob('*'):
        if file.name == 'description.txt'  :
            with open(file, 'r',encoding='utf-8') as f:
                description=f.read()
            # print(description)
            documents.append(Document(description, content_type='text'))
    if not description:
        print(f"Warning: No file named 'description.txt' found in directory {ab_dir_path}")

    # load yaml file

    pipeline_config.append(load_pipeline_config(config_path))

    if algo=='BO':
        print('Using Bayesian Optimization')
        pipeline_config.append(load_pipeline_config('./pipeline_bo.yaml'))

    p1= Pipeline()
    node_list=[]
    for config in pipeline_config[0]:
        node_list.append(initialize_node(config['node']))
    for node,config in zip(node_list,pipeline_config[0]):
        p1.add_node(node, config['name'], inputs=config.get('inputs', []))
    
    p1.run(query=query, documents=documents)
    
    # train_result,process_config=result['train_result'],result['process_config']
    p2=Pipeline()
    node_list=[]
    for config in pipeline_config[1]:
        node_list.append(initialize_node(config['node'],outputs=config.get('outputs', None)))
    for node,config in zip(node_list,pipeline_config[1]):

        p2.add_node(node, config['name'], inputs=config.get('inputs', []))

    from BO import AutoMLSpace
    num_features=len(documents[0].content.columns[:-1])
    op_dict={
           
            0:'feature_transformation',
            1:'feature_selection',
            2:'dimensionality_reduction',
            3:'resampling',


    }
    sp=AutoMLSpace(num_features=num_features,op_dict=op_dict)

    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    from hebo.optimizers.bo import BO
    import numpy as np


    def objective(params : pd.DataFrame) -> np.ndarray:
        result=p2.run(query='train',documents=['data.pkl',sp.decode_sample(params)])
        train_result=result['result']
        result=np.array([[train_result]])
        return -result  # Placeholder for actual computation


    opt   = HEBO(sp.space, rand_sample = 8)
    import tqdm
    with tqdm.tqdm(total=16) as pbar:
        for i in range(16):
            # print('#####',opt.y)
            try:
                rec = opt.suggest(n_suggestions = 1)
                y=objective(rec)
                y=np.array([y],dtype=np.float64)
                opt.observe(rec, y)
            except Exception as e:
                print(e)
                continue
            pbar.update(1)
            print('After %d iterations, best obj is %.2f' % (i, -opt.best_y))
    print("Best params is: \n",sp.decode_sample(opt.best_x))


    documents=[]
    for file in ab_dir_path.glob('*'):
        if file.name=='test.csv':
            data=pd.read_csv(file)
            data['target'] = 0
            data.to_pickle('test.pkl')
    p2.run(query='predict',documents=['test.pkl',sp.decode_sample(opt.best_x)])

if __name__ == '__main__':

    main()