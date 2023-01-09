import torch
import torch.nn as nn
import numpy as np
from utils import _transpose_and_gather_feat
import torch.nn.functional as F

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, pred, gt):
    return self.neg_loss(pred, gt)

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, gt):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class FocalLossCE(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self, alpha, gamma):
    super(FocalLossCE, self).__init__()
    self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0)
    self.alpha = alpha
    self.gamma = gamma
    
  def forward(self, pred, gt):
    ce_loss = self.cross_entropy(pred, gt) # important to add reduction='none' to keep per-batch-item loss
    pt = torch.exp(-ce_loss)
    focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean() # mean over the batch

    return focal_loss