# 
[ ! -d app/logs ] && mkdir app/logs
gpu=$1
if [ -z $gpu ]; then
    gpu=0,1,2
fi
export CUDA_VISIBLE_DEVICES=$gpu
export MKL_SERVICE_FORCE_INTEL=1
# export MKL_THREADING_LAYER = GNU

# datasets configuration with _ppl is designed for base model typically. configuration with _gen can be used for both base model and chat model.
model=hf_llama3_8b_instruct
nohup python run.py \
    --models $model \
    --datasets mmlu_gen \
    --work-dir app/outputs/$model \
    > app/logs/run.py-gpu$gpu.log 2>&1 &
