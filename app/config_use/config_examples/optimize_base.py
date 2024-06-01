# Configuration file writing
from torch.optim import SGD


optimizer = dict(type=SGD, lr=0.1)

x = dict(a=1, b=dict(c=2, d=3))