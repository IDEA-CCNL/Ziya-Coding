# Ziya-Coding-34B-v1.0

# 姜子牙系列模型

- [Ziya-LLaMA-13B-v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1)
- [Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
- [Ziya-LLaMA-7B-Reward](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward)
- [Ziya-LLaMA-13B-Pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1)
- [Ziya-BLIP2-14B-Visual-v1](https://huggingface.co/IDEA-CCNL/Ziya-BLIP2-14B-Visual-v1)
- [Ziya-Writing-LLaMa-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-Writing-LLaMa-13B-v1)
- [Ziya-Coding-15B-v1](https://huggingface.co/IDEA-CCNL/Ziya-Coding-15B-v1)

## 简介 Brief Introduction

使用自然语言生成高质量的代码是大模型落地中的高频需求。今天，IDEA研究院封神榜团队正式开源最新的代码大模型Ziya-Coding-34B-v1.0，我们在HumanEval Pass@1的评测上，取得了75.5的好成绩，超过了GPT-4（67.0）的得分，也成为目前已知开源模型新高。封神榜团队正在为社区提供先进的大模型技术和经验，帮助生产和定制更多优秀垂类模型，推进大模型生态发展。


Generating high-quality code using natural language is a high-frequency demand in the deployment of large models. Today, the IDEA Research Institute's Fengshenbang team officially open-sourced the latest code model, Ziya-Coding-34B-v1.0. We achieved a good score of 75.5 on the HumanEval Pass@1 evaluation, surpassing the score of GPT-4 (67.0) and setting a new high for known open-source models. The Fengshenbang team is providing the community with advanced large model technology and experience, helping to produce and customize more excellent vertical models, and promoting the development of the large model ecosystem.


更多细节可以参考我们的公众号文章：

[再创新高！姜子牙大模型开源代码大模型Ziya-Coding-34B-v1.0](https://mp.weixin.qq.com/s/Op4Wkiu2J9jwFr_Zj0YSZg)

[姜子牙大模型系列 | 代码模型ziya-coding发布！低成本微调即可学会在专有场景编程](https://mp.weixin.qq.com/s/tWaRF1wL3HM87ZDEawd2UA)

## 软件依赖
```
pip install torch==1.12.1 tokenizers==0.13.3 transformers==4.31.1
```

## 模型信息 Model Information

在9月初，我们开源了基于StarCoder-15B的代码模型Ziya-Coding-15B-v1，我们将训练Ziya-Coding-15B-v1积累的训练经验迁移到了新版本的训练中。

我们收集并构造了约45万涵盖了几乎所有代码相关任务的指令数据进行第一阶段的微调，这其中包括约10万的中文指令和35万的英文指令，保证了数据的多样性，在构造数据时，我们充分利用了高质量的无指令代码数据，使用LLM生成对应的指令，扩充得到了更多高质量的代码指令数据。

同时实验过程中，我们注意到，代码指令的难度和正确性是训练代码模型成功的关键。因此，我们引入了第二阶段的精调。我们使用evol-instruct的方法生成了大量高难度多要求的代码指令数据，并利用代码编译器作为反馈，筛选出能够通过编译的代码。最后利用LLM生成单元测试进一步验证代码的正确性。我们最终筛选出了46k数据，在第一阶段模型的基础上，使用较低的学习率进行微调，最终得到了我们的Ziya-coding-34B-v1.0。



In early September, we open-sourced the code model Ziya-Coding-15B-v1 based on StarCoder-15B. The training experience accumulated in training Ziya-Coding-15B-v1 was transferred to the training of the new version.


We collected and constructed about 450,000 instruction data covering almost all code-related tasks for the first stage of fine-tuning. This includes about 100,000 Chinese instructions and 350,000 English instructions, ensuring data diversity. When constructing the data, we made full use of high-quality non-instructional code data, used LLM to generate corresponding instructions, and expanded to obtain more high-quality code instruction data.

During the experiment, we noticed that the difficulty and correctness of code instructions are key to the successful training of code models. Therefore, we introduced a second stage of fine-tuning. We used the evol-instruct method to generate a large amount of high-difficulty, multi-requirement code instruction data, and used a code compiler as feedback to filter out code that could pass compilation. Finally, we used LLM to generate unit tests to further verify the correctness of the code. We ultimately filtered out 46k data, and on the basis of the first-stage model, we fine-tuned it with a lower learning rate to finally obtain our Ziya-coding-34B-v1.0.

### 效果评估 Performance

| Model                       | HumanEval(pass@1) | 
|:----------------------------|:-----------------:|
| **Ziya-Coding-34B-v1.0**  |     **75.5%**      |
| CodeFuse-CodeLlama-34B  |     74.4%     |
| Phind-CodeLLaMa-34B-v2 |       73.8%       | 
| WizardCoder-Python-34B-V1.0 |       73.2%       |
| GPT-4            |       67.0%       |
| PanGu-Coder2 15B            |       61.6%       |
| WizardCoder-15B-V1.0                   |       59.8%       |
| CodeLlama-34b-Python        |       53.7%       |
| Ziya-Coding-15B-v1                   |       50.1%       | 
| CodeLlama-34b               |       48.8%       |
| GPT-3.5          |       48.1%       |
| StarCoder-15B               |       33.6%       |


其中，我们对微调数据集进行了去污处理，避免数据泄露，HumanEval的pass@1指标是贪婪生成的结果。

Prompt Format
```python3
"<human>: \nPlease Complete the given function below according to the docstring: \n{prompt}\n<bot>: \n"
```
In this process, we performed a decontamination process on the fine-tuning dataset to avoid data leakage. The pass@1 metric for HumanEval is based on the results of greedy generation.

## <span id="jump"> 使用 Usage </span>
```python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda")
prompt = "写一段快速排序"
model = AutoModelForCausalLM.from_pretrained("IDEA-CCNL/Ziya-Coding-34B-v1.0", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Ziya-Coding-34B-v1.0", use_fast=False)
input = f"<human>: \n{prompt}\n<bot>: \n"
input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
generate_ids = model.generate(
            input_ids,
            max_new_tokens = 512, 
            do_sample = True, 
            top_p = 0.85, 
            temperature = 1.0,
            repetition_penalty = 1.0,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
            )
output = tokenizer.batch_decode(generate_ids)[0]
print(output)
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2210.08590)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2210.08590):

```text
@article{fengshenbang,
  author    = {Jiaxing Zhang and Ruyi Gan and Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

欢迎引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):
```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```