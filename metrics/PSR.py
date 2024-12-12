import os
import pickle
import numpy as np
import yaml


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


config_path = os.path.abspath(f'../configs/metrics.yaml')
config = load_config(config_path)
metric_threshold = 0.1318359375 # metric threshold to define whether the protection succeed after generation
metric = 'clip_iqac'

method = 'CAAT'
assert method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])
dir_suffix = config[method]['dir_suffix']
suffix_length = len(dir_suffix.split('/')) - 1
metric_path_for_gridpure= ''
metric_path = ''

for key_ in ['','-JPEG-70','-JPEG-50','-Resize-2','-Resize-4']:
    with open(f'{metric_path}'+key_,'rb') as f:
        results = pickle.load(f)
    reform_dict = {}
    for idx,p in enumerate(results['path']):
        id_key =  p.split('/')[-(6+suffix_length)] 
        if id_key not in reform_dict:
            reform_dict[id_key] = {}
            reform_dict[id_key][metric] = []
        reform_dict[id_key][metric].append(results[metric][idx])


    generation_fail_rate = 0
    for key in reform_dict:
        generation_fail_rate += (np.array(reform_dict[key][metric]) < metric_threshold).sum() / 16
    print(key_,'Protection Success Rate:',generation_fail_rate/50)

with open(f'{metric_path_for_gridpure}','rb') as f:
        results = pickle.load(f)
reform_dict = {}
for idx,p in enumerate(results['path']):
    id_key =  p.split('/')[-(6+suffix_length)] 
    if id_key not in reform_dict:
        reform_dict[id_key] = {}
        reform_dict[id_key][metric] = []
    reform_dict[id_key][metric].append(results[metric][idx])


generation_fail_rate = 0
for key in reform_dict:
    generation_fail_rate += (np.array(reform_dict[key][metric]) < metric_threshold).sum() / 16
print('GridPure:','Protection Success Rate:',generation_fail_rate/50)


