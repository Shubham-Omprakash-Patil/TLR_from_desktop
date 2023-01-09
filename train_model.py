import os
import torch
from torch.utils.data import DataLoader
import time
from progress.bar import Bar

import _init_paths

from lib.models import get_resnet, get_resnet_bifpn
from lib.losses import FocalLoss, RegL1Loss, FocalLossCE
from lib.utils import _sigmoid, AverageMeter
from lib.data_generation import MagnaAdTlDataset
from lib.opts import opts
from lib.logger import Logger
from lib.manager import save_model, load_model
from lib.trainer import Trainer

def main(opt):
    print(opt)
    
    # init logger
    logger = Logger(opt)

    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')  
    
    print("Using {} device".format(opt.device))
    
    # options
    learning_rate = opt.lr
    layers = opt.layers
    train_image_path = opt.train_image_path
    valid_image_path = opt.valid_image_path
    anno_path = opt.anno_path    
    batch_size = opt.batch_size    
    num_epochs = opt.num_epochs

    # model creation
    model = get_resnet(layers, opt.num_classes, opt.num_features).to(opt.device)
    # model = get_resnet_bifpn(layers, opt.num_classes, opt.num_features).to(opt.device)

    loss_hm = FocalLoss()
    loss_wh = torch.nn.L1Loss(reduction='sum') 
    loss_reg = RegL1Loss()        
    loss_cls = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_nb = torch.nn.CrossEntropyLoss(ignore_index=0)

    # allocate model to device
    if len(opt.gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=opt.gpus)
        loss_hm = torch.nn.DataParallel(loss_hm, device_ids=opt.gpus)
        loss_wh = torch.nn.DataParallel(loss_wh, device_ids=opt.gpus)
        loss_reg = torch.nn.DataParallel(loss_reg, device_ids=opt.gpus)
    
    loss_hm.to(device=opt.device)
    loss_wh.to(device=opt.device)
    loss_reg.to(device=opt.device)
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'cls_loss', 'nb_loss']
    losses = {'loss_hm': loss_hm,
              'loss_wh': loss_wh,
              'loss_reg': loss_reg,
              'loss_cls': loss_cls,
              'loss_nb': loss_nb}

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)    
    
    # init dataloader
    dataset = MagnaAdTlDataset(train_image_path, anno_path, aug=True, debug=False)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=48)
    dataset = MagnaAdTlDataset(valid_image_path, anno_path, aug=False, debug=False)
    valid_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=48)

    start_epoch = 0
    # load model
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)   

    trainer = Trainer(opt, model, losses, loss_states, optimizer)
    best = 1e10
    for epoch in range(start_epoch + 1, num_epochs + 1):        
        mark = epoch if opt.save_all else 'last'
        ret = trainer.train(epoch, train_dataloader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in ret.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, valid_dataloader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)

        logger.write('\n')        
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)            
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)