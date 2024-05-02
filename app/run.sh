# 
[ ! -d app/logs ] && mkdir app/logs
gpu=$1
if [ -z $gpu ]; then
    gpu=0,1
fi
export CUDA_VISIBLE_DEVICES=$gpu
export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER = GNU

model=hf_llama3_8b_instruct
nohup python run.py \
    --models $model \
    --datasets mmlu_ppl \
    --work-dir app/outputs/$model \
    > app/logs/run.py-gpu$gpu.log 2>&1 &

# model=hf_llama3_8b_instruct
# nohup python run.py --datasets mmlu_gen \
# --hf-path /mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct \
# --model-kwargs device_map='auto' \
# --tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \
# --max-out-len 200 \
# --max-seq-len 2048 \
# --batch-size 1 \
# --no-batch-padding \
# --num-gpus 1 \
# --work-dir app/outputs/$model \
# > app/logs/run.py-gpu$gpu.log 2>&1 &