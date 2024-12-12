import os
import pickle
import numpy as np
import yaml



metric_threshold = 0.1318359375 # metric threshold to define whether the protection succeed after generation
bit_error_threshold = 3/32 # bit error threshold to define whether the image is authorized
metric = 'clip_iqac'
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
config_path = os.path.abspath(f'../configs/metrics.yaml')
config = load_config(config_path)



method = 'CAAT'
assert method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])
print(method)

dir_suffix = config[method]['dir_suffix']
suffix_length = len(dir_suffix.split('/')) - 1

bit_error_path = ''
bit_error_path_for_gridpure = ''
metric_path = ''

### For rejection #######

with open(bit_error_path,'rb') as f:
    results_biterror = pickle.load(f)


for the_key in [ 'origin', 'JPEG_70', 'JPEG_50', 'Resize_2', 'Resize_4']:
    reform_dict = {}
    for idx,p in enumerate(results_biterror['path']):
        id_key =  p.split('/')[-(2+suffix_length)] 
        if id_key not in reform_dict:
            reform_dict[id_key] = {}

    for idx,p in enumerate(results_biterror['path']):
        id_key =  p.split('/')[-(2+suffix_length)] 
        if  'bit_error' not in reform_dict[id_key]:
            reform_dict[id_key]['bit_error'] = []
        reform_dict[id_key]['bit_error'].append(results_biterror[the_key][idx]) 


    unauthorized_rate = 0
    protection_success_rate = 0


    for key in reform_dict:
        unauthorized_rate += (np.array(reform_dict[key]['bit_error']) > bit_error_threshold).sum() / 4
        protection_success_rate += int((np.array(reform_dict[key]['bit_error']) > bit_error_threshold).sum() > 0 ) 
    assert (unauthorized_rate / 50) == 0 or (unauthorized_rate / 50) ==1
    print(the_key,':', 'Unauthorized Image Rate:',unauthorized_rate / 50, 'Protection Success Rate:',protection_success_rate / 50)
    

with open(bit_error_path_for_gridpure,'rb') as f:
    results_biterror = pickle.load(f)

reform_dict = {}
for idx,p in enumerate(results_biterror['path']):
    id_key =  p.split('/')[-2] 
    if id_key not in reform_dict:
        reform_dict[id_key] = {}

for idx,p in enumerate(results_biterror['path']):
    id_key =  p.split('/')[-2] 
    if  'bit_error' not in reform_dict[id_key]:
        reform_dict[id_key]['bit_error'] = []
    reform_dict[id_key]['bit_error'].append(results_biterror['pure'][idx]) 


unauthorized_rate = 0
protection_success_rate = 0

for key in reform_dict:
    unauthorized_rate += (np.array(reform_dict[key]['bit_error']) > bit_error_threshold).sum() / 4
    protection_success_rate += int((np.array(reform_dict[key]['bit_error']) > bit_error_threshold).sum() > 0 )
assert (unauthorized_rate / 50) == 0 or (unauthorized_rate / 50) ==1
print('GridPure:', 'Unauthorized Image Rate:',unauthorized_rate / 50, 'Protection Success Rate:',protection_success_rate / 50)

### For protection ####

### As In the experiment, all images become unauthorized after purification, we only need to calculate the Protection Success Rate under clean setting here.

with open(metric_path,'rb') as f:
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
print('Protection Success Rate:',generation_fail_rate/50)

