# 
[ ! -d app/logs ] && mkdir app/logs
gpu=$1
if [ -z $gpu ]; then
    gpu=1
fi
export CUDA_VISIBLE_DEVICES=$gpu

model=hf_llama3_8b_instruct.py
nohup python run.py \
    --models $model \
    --datasets mmlu_ppl \
    --work-dir app/outputs/$model \
    > app/logs/run.py-gpu$gpu.log 2>&1 &