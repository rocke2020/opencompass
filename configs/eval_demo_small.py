from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .models.opt.hf_opt_125m import opt125m

datasets = [*siqa_datasets]
models = [opt125m]
