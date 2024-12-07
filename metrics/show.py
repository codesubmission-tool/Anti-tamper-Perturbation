import pickle
import numpy as np
import argparse

def get_args():
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('--path', type=str, default=None)
    args = parent_parser.parse_args()
    return args

def get_ism(results):
    ism_result = np.array(results)
    ism_score = ism_result[ism_result!=-1].mean()
    fdfr_rate = (ism_result==-1).mean()
    return ism_score,fdfr_rate
#CLIP LIQE ISM FDFR

def print_results(path):
    with open(path,'rb') as f:
        results = pickle.load(f)
    ism_score,fdfr_rate = get_ism(results['ism'])
    print('%f	%f	%f	%f'%((results['clip_iqac'].mean()),results['liqe'].mean(),ism_score,fdfr_rate))

args = get_args()
locals().update(vars(args))
print('CLIP-IQAC	LIQE            ISM             FDFR')
print_results(path)