# The construction process is exactly the same
import torch.nn as nn
from mmengine.registry import OPTIMIZERS
from mmengine.config import Config

cfg = Config.fromfile('app/config_use/config_examples/optimizer.py')
model = nn.Conv2d(1, 1, 1)
cfg.optimizer.params = model.parameters()
optimizer = OPTIMIZERS.build(cfg.optimizer)