from glob import glob
import os
import yaml
import argparse


def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--method', type=str, default='CAAT')
    args = parent_parser.parse_args()
    return args

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

args = get_args()
locals().update(vars(args))

assert method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])

config_path = os.path.abspath('../configs/protection.yaml')
config = load_config(config_path)
print(config_path,config['instance_dir'])
instance_dir = config['instance_dir']
instance_dir = glob(f'{instance_dir}/*')
output_dir = config['output_dir']
os.makedirs(output_dir,exist_ok=True)

for key in config[method]:
    print(key,config[method][key])

if method == 'CAAT':
    for each in instance_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        os.system('cd CAAT; CUDA_VISIBLE_DEVICES=0 bash train_CAAT_freq.sh %s %s %s'%(each+'/set_B',output_dir+'/'+i_name,config_path))
        
elif method == 'METACLOAK':
    for each in instance_dir:
        i_name = each.split('/')[-1] 
        if '.pt' in i_name:
            continue
        print(i_name)
        os.system('cd CAAT; CUDA_VISIBLE_DEVICES=0 bash train_CAAT_freq.sh %s %s %s'%(each+'/set_B',output_dir+'/'+i_name,config_path))

else:
    raise "Method Unknown"