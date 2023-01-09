from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from utils import _sigmoid, AverageMeter

class Trainer(object):
    def __init__(self, opt, model, losses, loss_states, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.losses = losses
        self.loss_states = loss_states
        self.model = model
        #self.decay_rate = 0.96
        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.decay_rate)
  
    def run_epoch(self, phase, epoch, data_loader):
        model = self.model
        if phase == 'train':
            model.train()
        else:
            model.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        num_iters = len(data_loader)
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_states}
        bar = Bar('{}/{}'.format(phase, 'resnet_{}'.format(opt.layers)), max=num_iters)
        end = time.time()

        for iter_id, rets in enumerate(data_loader):
            data_time.update(time.time() - end)

            # Compute prediction error
            batch_input = []
            batch_hm = []
            batch_reg = []
            batch_reg_mask = []
            batch_ind = []
            batch_dense_wh = []
            batch_dense_wh_mask = []
            batch_cls = []
            batch_nb = []

            for ret in rets:
                batch_input.append(ret['input'])
                batch_hm.append(ret['hm'])
                batch_reg.append(ret['reg'])
                batch_reg_mask.append(ret['reg_mask'])
                batch_ind.append(ret['ind'])
                batch_dense_wh.append(ret['dense_wh'])
                batch_dense_wh_mask.append(ret['dense_wh_mask'])
                batch_cls.append(ret['cls'])
                batch_nb.append(ret['nb'])    

            batch_input = torch.cat(batch_input, dim=0).to(device=opt.device, non_blocking=True)
            batch_hm = torch.cat(batch_hm, dim=0).to(device=opt.device, non_blocking=True)
            batch_reg = torch.cat(batch_reg, dim=0).to(device=opt.device, non_blocking=True)
            batch_reg_mask = torch.cat(batch_reg_mask, dim=0).to(device=opt.device, non_blocking=True)
            batch_ind = torch.cat(batch_ind, dim=0).to(device=opt.device, non_blocking=True)
            batch_dense_wh = torch.cat(batch_dense_wh, dim=0).to(device=opt.device, non_blocking=True)
            batch_dense_wh_mask = torch.cat(batch_dense_wh_mask, dim=0).to(device=opt.device, non_blocking=True)
            batch_cls = torch.cat(batch_cls, dim=0).to(device=opt.device, non_blocking=True)
            batch_nb = torch.cat(batch_nb, dim=0).to(device=opt.device, non_blocking=True)

            #for k in batch:
            #    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            outputs = model(batch_input)
            outputs = outputs[0]

            hm_loss, wh_loss, off_loss, cls_loss, nb_loss = 0, 0, 0, 0, 0
            outputs['hm'] = _sigmoid(outputs['hm'])

            hm_loss += self.losses['loss_hm'](outputs['hm'], batch_hm)
            mask_weight = batch_dense_wh_mask.sum() + 1e-4
            wh_loss += self.losses['loss_wh'](outputs['wh'] * batch_dense_wh_mask,batch_dense_wh * batch_dense_wh_mask) / mask_weight
            off_loss += self.losses['loss_reg'](outputs['reg'], batch_reg_mask, batch_ind, batch_reg)
            #cls_loss += self.losses['loss_cls'](outputs['cls'] * batch_dense_wh_mask[:,1,:,:], batch_cls.long() * batch_dense_wh_mask[:,1,:,:])
            #nb_loss += self.losses['loss_nb'](outputs['nb'] * batch_dense_wh_mask[:,1,:,:], batch_nb.long() * batch_dense_wh_mask[:,1,:,:])
            cls_loss += self.losses['loss_cls'](outputs['cls'], batch_cls.long())
            nb_loss += self.losses['loss_nb'](outputs['nb'], batch_nb.long())

            loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.cls_weight * cls_loss + opt.nb_weight * nb_loss
            loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 'cls_loss': cls_loss, 'nb_loss': nb_loss}
            loss = loss.mean()

            # Backpropagation
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()      
                
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(epoch, iter_id+1, num_iters, phase=phase, total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch_input.size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)            
            bar.next()            
        
        bar.finish()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.

        return ret
  
    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)