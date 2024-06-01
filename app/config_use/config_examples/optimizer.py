from mmengine.config import read_base


with read_base():
    from .optimize_base import *

# optimizer is a variable defined in the base configuration file
# optimizer.update(
#     lr=0.01,
# )
x.update(dict(b=dict(d=4)))