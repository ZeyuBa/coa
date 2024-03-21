from uu import decode
from skopt.space import Integer
from itertools import combinations, permutations
import random
import numpy as np
from hebo.design_space.design_space import DesignSpace
class AutoMLSpace:
    def __init__(self, num_features, op_dict):
        self.num_features = num_features
        self.op_dict=op_dict
        self.num_operations = len(self.op_dict)
        self.operations = self._initialize_operations()
        self.space = self._create_search_space()


    def _initialize_operations(self):
        operations = {
            'feature_transformation': ['normalization', 'standardization']
                                +['N/A'],
            'feature_selection': list(self._calculate_feature_subsets().keys())
                                +['N/A'],
            'dimensionality_reduction': ['PCA']
                                #   +[ 'ISOMAP']
                                +['N/A']
                                ,
            'resampling': ['N/A', 'upsampling', 'downsampling'],
            'order': list(self._calculate_operation_orders().keys()),
            'n_components': list(range(1, self.num_features-1))
        }
        return operations

    def _calculate_feature_subsets(self):
        subsets = [combinations(range(self.num_features), i) for i in range(1, self.num_features + 1)]
        flattened = [item for sublist in subsets for item in sublist]
        flattened=sorted(flattened,key=lambda x:len(x),reverse=True)[1:50]
        # print(flattened)
        return {subset: index for index, subset in enumerate(flattened)}

    def _calculate_operation_orders(self):
        orders = list(permutations(range(self.num_operations)))
        return {order: index for index, order in enumerate(orders)}

    def _create_search_space(self):
        # params = [{'name': key, 'type':'cat', 'categories': [str(v) for v in values]} for key, values in self.operations.items()]
        params=[{'name': key, 'type':'int', 'lb' : 0, 'ub' : len(values) - 1} for key, values in self.operations.items()]
        # space = [Integer(0, len(values) - 1, name=key) for key, values in self.operations.items()]
        space = DesignSpace().parse(params)
        return space

    def sample_point(self):
        return self.space.sample()
        # return [dimension.rvs(random_state=random.randint(1, 100)) for dimension in self.space]

    def decode_sample(self, sample):
        ks=list(self.operations.keys())
        vs=sample[ks].values[0]
        decoded = {k:self.operations[k][v] for k,v in zip(ks,vs) }
        pipeline=[0 for _ in range(self.num_operations)]
        for i in decoded['order']:
            
            if decoded[self.op_dict[i]] != 'N/A':
                if self.op_dict[i] == 'dimensionality_reduction':
                    pipeline[i]=(self.op_dict[i],(decoded[self.op_dict[i]],decoded['n_components'])) 
                else: 
                     pipeline[i]=(self.op_dict[i],decoded[self.op_dict[i]],)
            # Apply the constraints
        feature_selection_idx = self.op_dict.keys().index('feature_selection') if 'feature_selection' in self.op_dict else -1
        dimensionality_reduction_idx = self.op_dict.keys().index('dimensionality_reduction') if 'dimensionality_reduction' in self.op_dict else -1

        if feature_selection_idx > dimensionality_reduction_idx and dimensionality_reduction_idx != -1 and feature_selection_idx != -1:
            pipeline[feature_selection_idx] = 0

        if 'feature_selection' in decoded and decoded['feature_selection'] != 'N/A' and decoded['dimensionality_reduction'] != 'N/A':
            selected_features_len = len(tuple(decoded['feature_selection']))
            if decoded['n_components'] > selected_features_len:
                pipeline[dimensionality_reduction_idx] = 0
        print(pipeline)
        return [i for i in pipeline if i ]

# Example usage
if __name__ == "__main__":
    num_features = 3
    op_dict={
           
            0:'feature_transformation',
            1:'feature_selection',
            2:'dimensionality_reduction',
            # 3:'resampling',


    }
    sp = AutoMLSpace(num_features, op_dict)

    # print(sp.operations['feature_selection'])
    sampled_point = sp.sample_point()
    # print(sampled_point)
    from hebo.optimizers.hebo import HEBO
    opt   = HEBO(sp.space, rand_sample = 4)
    for _ in range(5):
        rec = opt.suggest(n_suggestions = 1)
        # print(rec)

    x,xe = sp.space.transform(sampled_point)
    decoded_operations = sp.decode_sample(sampled_point)

    print("Sampled Point:", sampled_point)
    print("Decoded Operations:", decoded_operations)
