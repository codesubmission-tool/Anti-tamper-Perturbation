from glob import glob
import os
import argparse
import yaml

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--dataset', type=str, default='CelebA-HQ')
    parent_parser.add_argument('--method', type=str, default='CAAT')
    args = parent_parser.parse_args()
    return args


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

args = get_args()
locals().update(vars(args))
assert dataset in set(['VGGFac2','CelebA-HQ'])
assert method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])

config_path = os.path.abspath(f'../configs/eval_{dataset}.yaml')
config = load_config(config_path)
instance_dir = config['instance_dir']
instance_dir = glob(f'{instance_dir}/*')
output_dir = config['target_dir']
os.makedirs(output_dir,exist_ok=True)
dir_suffix = config[method]['dir_suffix']
for each in instance_dir:
    i_name = each.split('/')[-1]
    if('.pt'in i_name):
        continue
    print(i_name)
    os.system('CUDA_VISIBLE_DEVICES=0 bash scripts/train_DB.sh %s '%(output_dir+'/'+i_name+f'/{dir_suffix}'))
    # break
