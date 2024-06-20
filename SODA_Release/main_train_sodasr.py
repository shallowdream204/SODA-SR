from inspect import stack
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
from torchvision.transforms import RandomHorizontalFlip, RandomRotation


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',pretrained_path=opt['path']['pretrained_netG'])
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    origin_type = opt['netG']['net_type']
    opt['netG']['net_type'] = 'swinir_gumbel'
    model_teacher= define_Model(opt)
    model_teacher.load()
    model_teacher.netG.eval()
    for p in model_teacher.netG.parameters():
        p.requires_grad = False
    opt['netG']['net_type'] = origin_type

    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    RPF = RandomHorizontalFlip(p=1)
    R90 = RandomRotation((90,90))
    R180 = RandomRotation((180,180))
    R270 = RandomRotation((270,270))
    Rn90 = RandomRotation((-90,-90))
    Rn180 = RandomRotation((-180,-180))
    Rn270 = RandomRotation((-270,-270))

    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        for i, train_data in enumerate(train_loader):
            
            with torch.no_grad():
                pslabel_list = []
                for _ in range(5):
                    pslabel1 = RPF(model_teacher.netG(RPF(train_data['L'])))
                    pslabel2 = RPF(Rn90(model_teacher.netG(R90(RPF(train_data['L'])))))
                    pslabel3 = RPF(Rn180(model_teacher.netG(R180(RPF(train_data['L'])))))
                    pslabel4 = RPF(Rn270(model_teacher.netG(R270(RPF(train_data['L'])))))
                    pslabel5 = model_teacher.netG(train_data['L'])
                    pslabel6 = Rn90(model_teacher.netG(R90(train_data['L'])))
                    pslabel7 = Rn180(model_teacher.netG(R180(train_data['L'])))
                    pslabel8 = Rn270(model_teacher.netG(R270(train_data['L'])))
                    pslabel_list.append((pslabel1+pslabel2+pslabel3+pslabel4+pslabel5+pslabel6+pslabel7+pslabel8) / 8.)

                stacked_pslabel =torch.stack(pslabel_list)
                mea = torch.mean(stacked_pslabel, dim=0) 
                var = torch.var(stacked_pslabel, unbiased=False, dim=0)
                confidence = 3/2 - torch.sigmoid(var/ 0.0004)
                train_data['H'] = mea
             
            current_step += 1
            
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step,confidence)
            if current_step >= opt['train']['EMA_init']:
                teacher_params = dict(model_teacher.netG.module.named_parameters())
                student_params = dict(model.netG.module.named_parameters())
                for k in teacher_params.keys():
                    teacher_params[k].data.mul_(opt['train']['E_decay']).add_((student_params[k]), 
                                                                              alpha=1-opt['train']['E_decay'])

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
                model_teacher.save_network(model_teacher.save_dir, model_teacher.netG, 'teacher', current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            # if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

            #     avg_psnr = 0.0
            #     idx = 0

            #     for test_data in test_loader:
            #         idx += 1
            #         image_name_ext = os.path.basename(test_data['L_path'][0])
            #         img_name, ext = os.path.splitext(image_name_ext)

            #         img_dir = os.path.join(opt['path']['images'], img_name)
            #         util.mkdir(img_dir)

            #         model.feed_data(test_data)
            #         model.test()

            #         visuals = model.current_visuals()
            #         E_img = util.tensor2uint(visuals['E'])
            #         H_img = util.tensor2uint(visuals['H'])

            #         # -----------------------
            #         # save estimated image E
            #         # -----------------------
            #         save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            #         util.imsave(E_img, save_img_path)

            #         # -----------------------
            #         # calculate PSNR
            #         # -----------------------
            #         current_psnr = util.calculate_psnr(E_img, H_img, border=border)

            #         logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

            #         avg_psnr += current_psnr

            #     avg_psnr = avg_psnr / idx

            #     # testing log
            #     logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
