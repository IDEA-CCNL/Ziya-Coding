# Ziya-Coding-34B-v1.0模型推理


## 目录
- [模型下载](#模型下载)
- [环境配置](#环境配置)


## 模型下载

| Size | Hugging Face Repo |
| ---  | --- |
| 34B   | [IDEA-CCNL/Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0)  

## 环境配置

#### Inference 推理环境配置

推理环境的最低配置要求为:
- 4 x RTX3090 (24GB)；

使用vllm框架复现human_eval结果
```bash
# setup environment
conda create --name inference python=3.10
conda activate inference
pip install -r requirements.txt
pip install -e human-eval
# inference
bash run_inference.sh
# human_eval pass@k 
evaluate_functional_correctness human_eval_inference/temperature_{}_results.json
```

