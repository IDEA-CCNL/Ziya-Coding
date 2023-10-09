model_path="your_model_path" #下载的Ziya-Coding-34B-v1.0的模型地址
temperature=0
max_tokens=2048
gpu_ids=0,1,2,3
num_gpus=4

results_path=human_eval_inference
mkdir -p ${results_path}

echo 'results_path:'$results_path
echo 'model_path:'$model_path
echo 'temperature:'$temperature
echo 'max_tokens:'$max_tokens
echo 'gpu_ids:'$gpu_ids
echo 'num_gpus:'$num_gpus

CUDA_VISIBLE_DEVICES=$gpu_ids python human_eval_inference.py \
        --model_path ${model_path} \
        --temperature ${temperature} \
        --num_gpus ${num_gpus} \
        --max_tokens ${max_tokens} \
        --results_path ${results_path} \
