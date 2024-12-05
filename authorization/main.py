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
from tqdm import tqdm
logging.basicConfig(level=logging.INFO,
    format='%(message)s',
    handlers=[
        #logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
        logging.StreamHandler(sys.stdout)
    ])
utils.set_seed(1042)

# class Trainer:

#      def __init__(self) -> None:
#         super().__init__()

def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'],strict=False)
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'],strict=False)
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])

args = options.get_args()
locals().update(vars(args))

message_length = 32
epoch_num = 200
# device = 'cuda:0'
runs_folder = ''
# experiment_name = 'experimentB'

val_interval = 1000
save_interval = 1000


model = HIDNet(device,gamma=gamma,randomFlag=random_maskFlag,input_mask = input_mask,pixel_space=pixel_space)
val_dataloader = torch.utils.data.DataLoader(CelebADataset(), batch_size=10, shuffle=False)
train_dataloader = torch.utils.data.DataLoader(FFHQDataset(), batch_size=8, shuffle=True)


#make exp_dir
this_run_folder = utils.create_folder_for_run(runs_folder,experiment_name)
log_dir = this_run_folder
tb_logger = utils.TensorBoardLogger(log_dir=log_dir)
# torch.autograd.set_detect_anomaly(True)
global_step = 0
val_step = 0 

for _ in range(epoch_num):
    logging.info('Epoch Num: {}'.format(_))
    for batch_id, image in tqdm(enumerate(train_dataloader)):
        global_step += 1
        image = image.to(device)
        message = utils.get_phis(phi_dimension=message_length,batch_size=image.shape[0]).to(device)
        losses, (encoded_images, random_mask, decoded_messages) = model.train_one_batch([image,message])
        tb_logger.save_losses(losses, global_step)
        if global_step % val_interval == 0:
            val_losses = {}
            val_step += 1
            for batch_id, image in enumerate(val_dataloader):
                image = image.to(device)
                message = utils.get_phis(phi_dimension=message_length,batch_size=image.shape[0]).to(device)
                losses, (encoded_images, random_mask, decoded_messages) = model.val_one_batch([image,message])
                for key in losses:
                    if key not in val_losses:
                        val_losses[key] = 0
                    val_losses[key] += losses[key]
            for key in val_losses:    
                val_losses[key] /= (batch_id+1)
                # print(encoded_images.shape)
            tb_logger.save_val_losses(val_losses, val_step)
            tb_logger.save_images((encoded_images[:2,:,:,:]+1)/2,val_step)

        if global_step % save_interval == 0:
            utils.save_checkpoint(model,experiment_name=experiment_name, epoch=global_step,checkpoint_folder=os.path.join(this_run_folder,'checkpoints'))
            
tb_logger.writer.close()




          

        



        