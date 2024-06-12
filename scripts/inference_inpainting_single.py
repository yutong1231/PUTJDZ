import os
import sys
from numpy.lib.arraysetops import isin
import torch
import json
import cv2
import random
import time
import argparse
import numpy as np
import warnings
import copy
from torch.nn.functional import l1_loss, mse_loss
from torch.serialization import save
import torchvision
import glob
from PIL import Image
from collections import OrderedDict, defaultdict
from torch.utils.data import ConcatDataset, dataset

from image_synthesis.utils.io import load_yaml_config, load_dict_from_json, save_dict_to_json
from image_synthesis.utils.misc import get_all_file, get_all_subdir, instantiate_from_config
from image_synthesis.utils.cal_metrics import get_PSNR, get_mse_loss, get_l1_loss, get_SSIM, get_mae
from image_synthesis.modeling.build import build_model
from image_synthesis.utils.misc import format_seconds, merge_opts_to_config
from image_synthesis.distributed.launch import launch
from image_synthesis.distributed.distributed import get_rank, reduce_dict, synchronize, all_gather
from image_synthesis.utils.misc import get_model_parameters_info, get_model_buffer

def read_mask(mask,size):
        
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        mask = np.array(mask).astype(np.float32)
        mask = mask / 255.0
        
        h, w = mask.shape[0], mask.shape[1]
        if (h,w) != tuple(size):
            mask_inp = cv2.resize(mask, tuple(size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_inp = copy.deepcopy(mask)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0:1] # h, w, 1
            mask_inp = mask_inp[:, :, 0:1]
        else:
            mask = mask[:, :, np.newaxis]
            mask_inp = mask_inp[:, :, np.newaxis]

        mask_inp = torch.tensor(mask_inp).permute(2, 0, 1).bool() # 1, h, w
        mask = torch.tensor(mask).permute(2, 0, 1).bool() # 1, h, w
        return mask_inp

def read_image(image,size):
        image = image.convert('RGB')
        # if not mask.mode == "RGB":
        #     mask = mask.convert("RGB")
        image = np.array(image).astype(np.float32)
        h, w = image.shape[0], image.shape[1]
        if (h,w) != tuple(size):
            image_inp = cv2.resize(image, tuple(size), interpolation=cv2.INTER_LINEAR)
        else:
            image_inp = copy.deepcopy(image)
        image_inp = torch.tensor(image_inp).permute(2, 0, 1) # 3, h, w
        image = torch.tensor(image).permute(2, 0, 1) # 3, h, w
        return image_inp

def get_model(args=None, model_name='2020-11-09T13-33-36_faceshq_vqgan'):
    if os.path.isfile(model_name):
        # import pdb; pdb.set_trace()
        if model_name.endswith(('.pth', '.ckpt')):
            model_path = model_name
            config_path = os.path.join(os.path.dirname(model_name), '..', 'configs', 'config.yaml')
        elif model_name.endswith('.yaml'):
            config_path = model_name
            model_path = os.path.join(os.path.dirname(model_name), '..', 'checkpoint', 'last.pth')
        else:
            raise RuntimeError(model_name)
        
        if 'OUTPUT' in model_name: # pretrained model
            model_name = model_name.split(os.path.sep)[-3]
        else: # just give a config file, such as test_openai_dvae.yaml, which is no need to train, just test
            model_name = os.path.basename(config_path).replace('.yaml', '')
    else:
        model_path = os.path.join('OUTPUT', model_name, 'checkpoint', 'last.pth')
        config_path = os.path.join('OUTPUT', model_name, 'configs', 'config.yaml')

    args.model_path = model_path
    args.config_path = config_path

    config = load_yaml_config(config_path)
    # config = merge_opts_to_config(config, args.opts)
    model = build_model(config)
    model_parameters = get_model_parameters_info(model)
    # import pdb; pdb.set_trace()
    print(model_parameters)

    # if model_path.endswith('.pth') or model_path.endswith('.ckpt'):
    #     save_path = model_path.replace('.pth', '_parameters.json').replace('.ckpt', '_parameters.json')
    #     json.dump(model_parameters, open(save_path, 'w'), indent=4)
    # sys.exit(1)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
    else:
        ckpt = {}
    if 'last_epoch' in ckpt:
        epoch = ckpt['last_epoch']
    elif 'epoch' in ckpt:
        epoch = ckpt['epoch']
    else:
        epoch = 0

    if 'model' in ckpt:
        # #TODO
        # # import pdb; pdb.set_trace()
        # model_static = OrderedDict()
        # for k in ckpt['model'].keys():
        #     if k.startswith('content_codec.'):
        #         # del ckpt['model'][k]
        #         print('delet: {}'.format(k))
        #     else:
        #         model_static[k] = ckpt['model'][k]
        # ckpt['model'] = model_static

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    elif 'state_dict' in ckpt:

        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        missing, unexpected = [], []
        print("====> Warning! No pretrained model!")
    print('Missing keys in created model:\n', missing)
    print('Unexpected keys in state dict:\n', unexpected)
    # import pdb; pdb.set_trace()

    model = model.eval()

    return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}




def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_dir', type=str, default='', 
                        help='directory to save results') 

    parser.add_argument('--name', type=str, default='transformer_JDG_white_mask', 
                        help='the name of this experiment, if not provided, set to'
                            'the name of config file') 
    parser.add_argument('--func', type=str, default='inference_inpainting', 
                        help='the name of inference function')

    parser.add_argument('--input_res', type=str, default=(256,256), 
                        help='input resolution (h,w)')
    parser.add_argument('--image_dir', type=str, default='',
                        help='gt images need to be loaded') 
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='mask dirs for image completion, Each gt image should have'
                        'a correspoding mask to be loaded')   

    
    # args for sampling
    parser.add_argument('--num_token_per_iter', type=str, default='10', 
                        help='the number of patches to be inpainted in one iteration')
    parser.add_argument('--num_token_for_sampling', type=str, default='100', 
                        help='the top-k tokens remained for sampling for each patch')
    parser.add_argument('--save_masked_image', action='store_true', default=False,    
                        help='Save the masked image, i.e., the input')
    parser.add_argument('--raster_order', action='store_true', default=False,    
                        help='Get the k1 patches in a raster order')

    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='auto', 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--num_replicate', type=int, default=1,
                        help='replaicate the batch data while forwarding. This may accelerate the sampling speed if num_sample > 1')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='The number of inpatined results to get for each image')

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    return args

def inference_inpainting(local_rank=0, args=None,img=None,mask=None):
    
    args=get_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node
    args.distributed = args.world_size > 1

    info = get_model(args=args, model_name=args.name)
    model = info['model']
    model = model.cuda()

    epoch = info['epoch']
    model_name = info['model_name']

    mask=read_mask(mask,args.input_res)
    img=read_image(img,args.input_res)
    data={'image':img,'mask':mask}
    filter_ratio = [float(fr) if '.' in fr else int(fr) for fr in args.num_token_for_sampling.split(',')] # [200] # [40, 50, 100]
    num_token_per_iter = [int(ntp) if '_' not in ntp else ntp for ntp in args.num_token_per_iter.split(',')] #[1, 20, 'average_50', 'cosine_50', 'linear_50']

    accumulate_time = None


    for fr in filter_ratio:
        for ntp in num_token_per_iter:
            
            # make a batch
            
            data['image'] = img.unsqueeze(dim=0)
            data['mask'] = mask.unsqueeze(dim=0)

            # import pdb; pdb.set_trace()
            
            # generate samples in a batch manner
            
            count_per_cond_ = 0
            
            while count_per_cond_ < args.num_sample:
                start_batch = time.time()
                with torch.no_grad():
                    content_dict = model.generate_content(
                        batch=copy.deepcopy(data),
                        filter_ratio=fr,
                        filter_type='count',
                        replicate=1 if args.num_sample == 1 else args.num_replicate,
                        with_process_bar=True,
                        mask_low_to_high=False,
                        sample_largest=True,
                        calculate_acc_and_prob=False,
                        num_token_per_iter=ntp,
                        accumulate_time=accumulate_time,
                        raster_order=args.raster_order
                    ) # B x C x H x W
                accumulate_time = content_dict.get('accumulate_time', None)
                print('Time consmption: ', accumulate_time)
                # save results
                for k in content_dict.keys():
                    # import pdb; pdb.set_trace()
                    if k in ['completed']:
                        content = content_dict[k].permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                        for b in range(content.shape[0]):
                            
                            im = Image.fromarray(content[b])
                            
                
                # prepare for next iteration
                print('==> batch time {}s'.format(round(time.time() - start_batch, 1)))
                if args.num_sample > 1:
                    count_per_cond_ = len(glob.glob(os.path.join(save_root_, '*completed*.png')))
                else:
                    count_per_cond_ += 1
    return im


if __name__=='__main__':
    image=Image.open('/disks/sda/yutong2333/PUT-main/data/JDG/test_img/3.jpg')
    mask=Image.open('/disks/sda/yutong2333/PUT-main/data/JDG/test_mask/3.png')
    processed_img=inference_inpainting(local_rank=0, args=None,img=image,mask=mask)
    processed_img.save('/disks/sda/yutong2333/PUT-main/3_processed.jpg')

    
    

    
