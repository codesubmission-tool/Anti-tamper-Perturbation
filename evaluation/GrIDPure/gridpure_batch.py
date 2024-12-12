import argparse
import torch
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
import sys
import os, yaml
import torch.nn as nn
from types import SimpleNamespace
from runners.diffpure_guided import GuidedDiffusion
from diffusers.utils import check_min_version
from torchvision import transforms
from glob import glob
check_min_version("0.15.0.dev0")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SDE_Adv_Model(nn.Module):
    # Adapted from https://github.com/NVlabs/DiffPure
    def __init__(self, args, config, model_dir):
        super().__init__()
        self.args = args
        self.runner = GuidedDiffusion(args, config, device=config.device, model_dir=model_dir)
        self.device = config.device
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=self.device)

    def set_tag(self, tag=None):
        self.tag = tag
    
    def set_pure_steps(self, pure_steps):
        self.args.t = pure_steps

    def forward(self, x):
        counter = self.counter.item()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        out = (x_re + 1) * 0.5
        self.counter += 1
        return out


class GrIDPure:
    def __init__(self, config_file, args, model_dir, pure_steps, pure_iter_num, gamma):
        super().__init__()
        self.pure_steps = pure_steps
        self.pure_iter_num = pure_iter_num
        self.gamma = gamma
        self.config_file = config_file
        self.args = args
        self.model_dir = model_dir
        self.model = self.creat_purimodel()
        self.transform = transforms.ToTensor()
        self.transform_back = transforms.ToPILImage()
        self.box_list = self.get_crop_box(512, 512)
        self.corner_positions = self.get_corner_box(512, 512)
        self.corner_rearrange_positions = [(0,0,128,128), (128,0,256,128), (0,128,128,256), (128,128,256,256)]

    def creat_purimodel(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config = SimpleNamespace(**self.config)
        self.config.device = device
        model = SDE_Adv_Model(self.args, self.config, self.model_dir)
        return model

    def get_crop_box(self, resolution_x, resolution_y):
        resolution_sub=128
        left_up_coor_x = []
        for coor_x in range(resolution_x):
            if (coor_x-256)%128==0 and coor_x<=resolution_x-256:
                left_up_coor_x.append(coor_x)
        if (resolution_x-256) not in left_up_coor_x:
            left_up_coor_x.append(resolution_x-256)
        left_up_coor_y = []
        for coor_y in range(resolution_y):
            if (coor_y-256)%128==0 and coor_y<=resolution_y-256:
                left_up_coor_y.append(coor_y)
        if (resolution_y-256) not in left_up_coor_y:
            left_up_coor_y.append(resolution_y-256)
        box_list = []
        for y in left_up_coor_y:
            for x in left_up_coor_x:
                box_list.append((x, y, x+256, y+256))
        return box_list

    def get_corner_box(self, resolution_x, resolution_y):
        return [(0,0,128,128), (resolution_x-128,0,resolution_x,128), (0,resolution_y-128,128,resolution_y), (resolution_x-128,resolution_y-128,resolution_x,resolution_y)]

    def merge_imgs(self, device,batch_size,img_list, corner_purified_imgs, positions, corner_positions, resolution_x, resolution_y):
        img_merged = torch.zeros([batch_size,3,resolution_x,resolution_y],dtype=float).to(device)
        merge_ind = torch.zeros([batch_size,3,resolution_x,resolution_y],dtype=float).to(device)
        # print(img_list.shape,corner_purified_imgs.shape)
        # img_list = [torch.tensor(img.cpu().numpy().transpose((0, 1, 3, 2))) for img in img_list]
        # corner_purified_imgs = torch.tensor(corner_purified_imgs.cpu().numpy().transpose((0, 1, 3, 2)))
        corner_rearrange_positions = [(0,0,128,128), (128,0,256,128), (0,128,128,256), (128,128,256,256)]
        for b_id in range(batch_size):
            for i, position in enumerate(positions):
                # print(b_id*batch_size+i)
                img_merged[b_id,:, position[0]:position[2], position[1]:position[3]] += img_list[b_id*len(positions)+i]
                merge_ind[b_id,:, position[0]:position[2], position[1]:position[3]] += torch.ones_like(img_list[b_id*len(positions)+i])
        for b_id in range(batch_size):
            for i in range(4):
                img_merged[b_id,:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]] += corner_purified_imgs[b_id][:,corner_rearrange_positions[i][0]:corner_rearrange_positions[i][2], corner_rearrange_positions[i][1]:corner_rearrange_positions[i][3]]
                merge_ind[b_id,:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]] += torch.ones_like(merge_ind[b_id,:, corner_positions[i][0]:corner_positions[i][2], corner_positions[i][1]:corner_positions[i][3]])
        img_merged = img_merged/merge_ind
        return img_merged.permute(0,1, 3, 2)

    def grid_pure(self, init_images):

        self.model = self.model.eval().to(self.config.device)
        self.model.set_pure_steps(self.pure_steps)
        batch_size=len(init_images)
        device = 'cuda:0'
        import time
        for pure_iter_idx in range(self.pure_iter_num):
            img_list = []
            corner_img_list = []
            for init_image in init_images:
                for box in self.box_list:
                    img_list.append(self.transform(init_image.crop(box)).to(device))
                corner_imgs = [init_image.crop(co) for co in self.corner_positions]
                corner_img = Image.new('RGB', (256, 256))
                for co_idx in range(4):
                    corner_img.paste(corner_imgs[co_idx], self.corner_rearrange_positions[co_idx])
                corner_img_list.append(self.transform(corner_img).to(device))
            img_list = torch.stack(img_list)
            start = time.time()
            corner_img_list = torch.stack(corner_img_list)
            img_list_pure = self.model(img_list).permute(0,1,3,2)
            corner_img_list_pure = self.model(corner_img_list).permute(0,1,3,2)
            imgs_pure = self.merge_imgs(device,batch_size,img_list_pure, corner_img_list_pure, self.box_list, self.corner_positions, 512,512)
            init_images_tmp = []
            for img_id,img_pure in enumerate(imgs_pure):
                img_pure = (1 - self.gamma) * img_pure + self.gamma * self.transform(init_images[img_id]).to(device)
                init_image = self.transform_back(img_pure.cpu())
                init_images_tmp.append(init_image)
            del init_images
            init_images = init_images_tmp
        return init_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrIDPure')
    parser.add_argument('--input_dir', type=str, help="path of images for purification")
    parser.add_argument('--output_dir', type=str, help="path of images for saving")
    parser.add_argument('--config_file', type=str, default="./imagenet.yml")
    parser.add_argument('--pure_model_dir', type=str, default=".")
    parser.add_argument('--pure_steps', type=int, default=100, help="purify steps")
    parser.add_argument('--pure_iter_num', type=int, default=1, help = "purify iter nums")
    parser.add_argument('--gamma', type=float, default=0.1, help = "gamma for blending")
    parser.add_argument('--method',  type=str, default="CAAT")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    diffpure_config_file = args.config_file
    diffpure_model_dir = args.pure_model_dir

    diffpure_args = {}
    diffpure_args['config'] = diffpure_config_file
    diffpure_args['data_seed'] = 0
    diffpure_args['seed'] = 1234
    diffpure_args['exp'] = 'exp'
    diffpure_args['verbose'] = 'info'
    diffpure_args['image_folder'] = 'images'
    diffpure_args['ni'] = False
    diffpure_args['sample_step'] = 1
    diffpure_args['t'] = 200
    diffpure_args['t_delta'] = 15
    diffpure_args['rand_t'] = False
    diffpure_args['diffusion_type'] = 'ddpm'
    diffpure_args['score_type'] = 'guided_diffusion'
    diffpure_args['eot_iter'] = 20
    diffpure_args['use_bm'] = False
    diffpure_args['sigma2'] = 1e-3
    diffpure_args['lambda_ld'] = 1e-2
    diffpure_args['eta'] = 5.
    diffpure_args['step_size'] = 1e-3
    diffpure_args['num_sub'] = 1000
    diffpure_args['adv_eps'] = 0.07
    diffpure_args = SimpleNamespace(**diffpure_args)
    

    




    def load_config(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    assert args.method in set(['CAAT','ANTIDB', 'ADVDM' ,'METACLOAK'])
    config_path = os.path.abspath(f'../../configs/eval_CelebA-HQ.yaml')
    config = load_config(config_path)
    dir_suffix = config[args.method]['dir_suffix']
    img_prefix = config[args.method]['img_prefix']
    batch_size = 2
    gridpure = GrIDPure(config_file=diffpure_config_file, 
                        args=diffpure_args,
                        model_dir=diffpure_model_dir, 
                        pure_steps=args.pure_steps, 
                        pure_iter_num=args.pure_iter_num,
                        gamma=args.gamma
                        )

    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    img_file_list = glob(f'{input_dir}/*/{dir_suffix}/{img_prefix}*')
    init_images = []
    file_path = []
    for img_idx, img_file in tqdm(enumerate(img_file_list), total=len(img_file_list)):
        init_image = Image.open(img_file).convert("RGB")
        init_images.append(init_image)
        file_path.append(img_file)
        

    for i in range(len(img_file_list)//batch_size):
        imgs = gridpure.grid_pure(init_images[(i)*batch_size:(i+1)*batch_size])
        for id_, file in enumerate(file_path[(i)*batch_size:(i+1)*batch_size]):
            name = file.split('/')[-1]
            id_name = file.split('/')[-3]
            print(id_name)
            os.makedirs(output_dir + '/{}'.format(id_name),exist_ok=True)
            img_file_name = output_dir + '/{}/{}'.format(id_name,name)
            imgs[id_].save(img_file_name)

    if len(img_file_list)%batch_size != 0:
        print(i)
        imgs = gridpure.grid_pure(init_images[(i+1)*batch_size:])  
        for id_, file in enumerate(file_path[(i+1)*batch_size:]):
            name = file.split('/')[-1]
            id_name = file.split('/')[-3]
            os.makedirs(output_dir + '/{}'.format(id_name),exist_ok=True)
            img_file_name = output_dir + '/{}/{}'.format(id_name,name)
            imgs[id_].save(img_file_name)
            # img_file_name = output_dir + '/{}.png'.format(img_idx)
            # img.save(img_file_name)

