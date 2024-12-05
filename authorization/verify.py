import os
import time
import torch
import numpy as np
import utils
import logging
from networks.HIDNet import HIDNet
import options
from dataset import *
import sys
import re
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import yaml

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file

def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'],strict=False)
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'],strict=False)
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('../configs/authorization.yaml')

logging.basicConfig(level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

seed = config['seed']
utils.set_seed(seed)
message_length = config['message_length']
run_folder = config['test']['run_folder']
store_folder = config['test']['store_folder']


print('seed:', seed)
print('message_length:', message_length)
print('running_folder:', os.path.join(run_folder,store_folder))
args = options.get_args()
locals().update(vars(args))


model = HIDNet(device,gamma=gamma,randomFlag=random_maskFlag,input_mask=input_mask,pixel_space=pixel_space)

dataset = config['test']['dataset']
assert dataset=='CelebA-HQ' or dataset=='VGGFace2'
if dataset == 'CelebA-HQ':
    val_dataloader = torch.utils.data.DataLoader(CelebADataset(train=False,get_path=True,path=config['data']['test_data_path']), batch_size=10, shuffle=False)
else:
    val_dataloader = torch.utils.data.DataLoader(VGGFaceDataset(train=False,get_path=True,path=config['data']['test_data_path']), batch_size=10, shuffle=False)

if path is not None:
    model_from_checkpoint(model,torch.load(path,map_location=device))

###########Authorization###########


val_losses = {}
testdir = os.path.join(run_folder,store_folder)
transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

for batch_id, (path,image) in tqdm(enumerate(val_dataloader)):
    image = image.to(device)
    message = utils.get_phis(phi_dimension=message_length,batch_size=image.shape[0]).to(device)
    losses, (encoded_images, random_mask, decoded_messages) = model.val_one_batch([image,message])
    # print('bitwise-error:',losses['bitwise-error  '])
    for key in losses:
        if key not in val_losses:
            val_losses[key] = 0
        val_losses[key] += losses[key]
    
    for id, e_img in enumerate(encoded_images):
        # print(id)
        image_name = path[id].split('/')[-1].split('.')[0]
        id_dir = path[id].split('/')[-3]
        set_dir = path[id].split('/')[-2]
        dir_path = os.path.join(testdir,id_dir,set_dir)
        os.makedirs(dir_path,exist_ok=True)
        transforms.ToPILImage()(((e_img.clip(-1,1)+1)/2).clip(0,1).cpu()).save('%s/%s.png'%(dir_path,image_name))
        # print(image_name)
        torch_set ={'random_mask':random_mask.cpu(),'message':message[id].cpu()}
        torch.save(torch_set,'%s/%s.pt'%(testdir,image_name))
        
###########Verification###########
random_mask = model.random_mask
model = model.encoder_decoder
model.eval()
bitwise_avg_err = []

paths = glob(f'{testdir}/*/set_B/*.png')

# meaningless_suffix = "noise-ckpt/50"



transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

test_set = testdir
print('noise_type',noise_type)
print('noise_args',noise_args)

if noise_type is not None:
    degrader = Degradation(noise_type=noise_type,noise_args=noise_args)
for path in tqdm(paths):
    # print(path)
    image_path = path
    if noise_type is None:
        image = transform(Image.open(image_path)).unsqueeze(0)
    else:
        image = cv2.imread(image_path).astype(np.float32)/255.
        image = degrader.degrade(image).unsqueeze(0)
    image = image.to(device)
    image_name = image_path.split('/')[-1]
    torch_set = torch.load(os.path.join(test_set,image_name.replace('.png','.pt').replace('50_noise_','')))
    message = torch_set['message'].to(device)
    random_mask = torch_set['random_mask'].to(device)
    img_dct_masked, img_dct, random_mask = model.dctlayer(image,random_mask=random_mask)
    encoded_dct, encoded_image = model.idctlayer(img_dct_masked,img_dct,random_mask)
    decoded_message = model.decoder(encoded_dct)
    decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
    bitwise_avg_err.append(np.sum(np.abs(decoded_rounded - message.round().detach().cpu().numpy())) / (
            message.shape[0]))


print('Average Bit-Error:',np.array(bitwise_avg_err).mean())





          

        



        