from vllm import LLM, SamplingParams

import os
import time 
import json
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
           
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help="")
parser.add_argument('--temperature', type=float, default=0, help="")
parser.add_argument('--results_path', type=str, help="")
parser.add_argument('--max_tokens', type=int, default=2048, help="")
parser.add_argument('--num_gpus', type=int, default=4, help="")
args = parser.parse_args()


sampling_params = SamplingParams(temperature=args.temperature, top_p=1, max_tokens=args.max_tokens, stop="</s>")
llm = LLM(model=args.model_path, tensor_parallel_size=args.num_gpus)


problems = read_problems()
task_ids = list(problems.keys())
prompts = [problems[task_id]['prompt'] for task_id in task_ids]


results = []
start_time = time.time()     
for idx, prompt in tqdm(enumerate(prompts[:1])):

    input = f"<human>: \nPlease Complete the given function below according to the docstring: \n{prompt}\n<bot>: \n"
    if idx==0:
        print("input", input)
    inputs = [input] * 200

    cur_time = time.time()
    outputs = llm.generate(inputs, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append({"task_id": task_ids[idx], "completion": generated_text})
    print(time.time() - cur_time)

print("total_time:", time.time() - start_time)

args.results_path = os.path.join(args.results_path, "temperature_{}_results.json".format(args.temperature))

with open(args.results_path, "w") as fout:
    for result in results:
        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
