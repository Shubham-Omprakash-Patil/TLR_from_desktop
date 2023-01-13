from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('--dataset', default='magna',
                             help='name of dataset') 
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
    self.parser.add_argument('--record', action='store_true',
                             help='for video recording')
    
    # system
    self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    self.parser.add_argument('--debugger_theme', default='black', 
                             choices=['white', 'black'])
    # model
    self.parser.add_argument('--layers', type=int, default=18, 
                             choices=[18, 34, 50, 101, 152])
    self.parser.add_argument('--num_classes', type=int, default=2,
                             help='number of classes.') 
    self.parser.add_argument('--num_features', type=int, default=16,
                             help='number of base feature.') 
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')

    # dataset
    self.parser.add_argument('--train_image_path', type=str, nargs='+', default=['/data/training/magna2019_tlr/train.json', '/data/training/magna2022_tlr_synthetic/train.json'],
                             help='path to image')
    self.parser.add_argument('--valid_image_path', type=str, nargs='+', default=['/data/training/magna2019_tlr/valid.json', '/data/training/magna2022_tlr_synthetic/valid.json'],
                             help='path to image')
    self.parser.add_argument('--test_image_path', type=str, nargs='+', default=['/data/training/magna2019_tlr/test.json'],
                             help='path to image')                              
    self.parser.add_argument('--db_root', type=str, nargs='+', default=['/mnt/magna-ad-db/Magna2019Database', '/mnt/magna-ad-db/magna2022synthetic/Dataset'],
                             help='path to root folder of dataset')    

    # 2D detection
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--cls_weight', type=float, default=1,
                             help='loss weight for cls classification.')
    self.parser.add_argument('--nb_weight', type=float, default=1,
                             help='loss weight for number of bulb classification.')

    # train
    self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
    self.parser.add_argument('--lr_step', type=str, default='90,120',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=140,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=8,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--gamma', type=float, default=2,
                             help='tuning parameter in focal loss')
                             
    # test
    self.parser.add_argument('--flip_test', action='store_true',
                             help='flip data augmentation.')
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=10,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')
    self.parser.add_argument('--fix_res', action='store_true',
                             help='fix testing resolution or keep '
                                  'the original resolution')
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')

    # exdet
    self.parser.add_argument('--agnostic_ex', action='store_true',
                             help='use category agnostic extreme points.')
    self.parser.add_argument('--scores_thresh', type=float, default=0.1,
                             help='threshold for extreme point heatmap.')
    self.parser.add_argument('--center_thresh', type=float, default=0.1,
                             help='threshold for centermap.')
    self.parser.add_argument('--aggr_weight', type=float, default=0.0,
                             help='edge aggregation weight.')

  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.root_dir = os.path.join(os.path.dirname(__file__))
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, '../exp')
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    opt.mean = [0.408, 0.447, 0.470]
    opt.std = [0.289, 0.274, 0.278]
    opt.pad = 31
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]

    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    
    return opt

  def init(self, args=''):    
    opt = self.parse(args)
    
    return opt
