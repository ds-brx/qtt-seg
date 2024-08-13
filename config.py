'''
1. Initialise space of configs by num_configs
2. corresponding curves of shape budget_len

'''

import json
import random
import uuid 


def sample_configurations(hp_file,num_configs,max_budget):
    
    with open(hp_file, 'r') as file:
        data = json.load(file)
    
    sequences = [param['sequence'] for param in data['HP']]
    names = [param['name'] for param in data['HP']]
    samples = []
    
    for _ in range(num_configs):
        config_id = str(uuid.uuid4())
        sampled_config  = {}
        sampled_config['idx'] = config_id
        sampled_config['config'] = {name: random.choice(sequence) for name, sequence in zip(names, sequences)}
        sampled_config['perf_curve'] = [0] * max_budget
        sampled_config['cost_curve'] = [0] * max_budget
        sampled_config["meta_feat"] = [5000, 19, 2, 3]
        
        samples.append(sampled_config)
    
    return samples




