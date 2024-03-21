# import importlib
# saved_code=['feature_selection.py']
# for file_path in saved_code:
#     op_name=file_path.split(".")[0]
#     try:
#         module= importlib.import_module("src."+op_name)  
#         importlib.reload(module)
#     except BaseException as e:
#         print('Cannot load module!\n')
    
#     try: 
#         func = getattr(module, op_name)
#     except AttributeError as e:
#         print('Cannot load module!\n')

    
#     if 'model' not  in op_name:
#         result=func
#     else: result=''

# print(result)
from itertools import combinations

def calculate_subsets(num_features):
    # Calculate all subsets of features
    subsets = []
    for i in range(1, num_features + 1):
        subsets.extend(list(combinations(range(num_features), i)))

    # Convert each subset to an integer and create a dictionary
    subset_dict = {}
    for subset in subsets:
        binary_str = ''.join(['1' if i in subset else '0' for i in range(num_features)])
        subset_integer = int(binary_str, 2)
        subset_dict[subset] = subset_integer

    return subset_dict



from itertools import permutations

def calculate_orders(num_operations):
    # Calculate all order combinations of operations
    orders = list(permutations(range(num_operations)))

    # Correspond each order to an integer
    order_to_int = {}
    for i, order in enumerate(orders):
        int_representation = i
        order_to_int[order] = int_representation

    return order_to_int

print(calculate_subsets(4))