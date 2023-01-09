import torch
import _init_paths
from lib.opts import opts

from lib.models import get_resnet, get_resnet_bifpn
from torchsummary import summary
from lib.manager import load_model
import time

# model = get_resnet(18, 2, 16).eval().cuda()
# summary(model, (3, 160, 320))

model = get_resnet_bifpn(18,2,16).eval().cuda()
summary(model, (3, 160, 320))

x = torch.rand((8,3,160, 320)).cuda()

# warm-up
for i in range(10):
    y = model(x)

t = 0
for i in range(100):
    s_t = time.time()
    y = model(x)
    dt = time.time() - s_t
    t += dt

print(t/100.*1000)

#  8: batch 1 - 2.8 | batch 8 - 2.9 - params: 250,039
# 12: batch 1 -     | batch 8 -     - params: 540,273
# 16: batch 1 -     | batch 8 - 3.3 - params: 940,055
# 32: batch 1 - 2.9 | batch 8 - 4.8 - params: 3,640,279
# 64: batch 1 - 2.9 | batch 8 - 9.1
'''
==============================================================
Total params: 711,544
Trainable params: 711,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 35592930852909.02
Params size (MB): 2.71
Estimated Total Size (MB): 35592930852912.31
----------------------------------------------------------------
'''