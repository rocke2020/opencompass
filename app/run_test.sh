# 
gpu=$1
if [ -z $gpu ]; then
    gpu=2
fi
export CUDA_VISIBLE_DEVICES=$gpu
export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER=GNU
file=run.py

# python $file configs/eval_demo.py \
# python $file \
#     --models opt125m \
#     --datasets siqa \
#     2>&1  </dev/null | tee app/logs/demo-$file.log

# 
nohup python $file configs/eval_demo_small.py \
    -w app/outputs/demo \
    --debug \
    > app/logs/demo-$file-gpu$gpu.log 2>&1 &