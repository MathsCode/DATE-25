import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm


# path = "/share/datasets/public_models/mistralai_Mixtral-8x7B-Instruct-v0.1/"
# path = "/home/xujiaming/xujiaming/train_machine/dataset/TurboSparse-Mistral-Instruct"
path = "/home/xujiaming/xujiaming/train_machine/dataset/TurboSparse-Mixtral"
dataset_name = ["ChatGPT-prompts", "Alpaca"][1]
datasets_path = "/share/xujiaming/datasets/Benchmark/"

ds = load_dataset(datasets_path + dataset_name)

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map='auto', trust_remote_code=True)


if dataset_name == "ChatGPT-prompts":
    dataset = ds["train"]
    for data in dataset:
        start = time.time()
        input = data['human_prompt']
        label = data['chatgpt_response']
        tokens = tokenizer(input, return_tensors='pt')
        generated_ids = model.generate(
            tokens.input_ids.to('cuda'),
            max_new_tokens=len(label)+1, 
            do_sample=True,
        )
        result = tokenizer.decode(generated_ids[0].tolist())
        with open("./tests/output/ChatGPT-prompt", mode='a') as f:
            print(label + '\n------------------------------\n' + result + '\n\n' + str(time.time()-start) + '\n------------------------------\n', file=f)

elif dataset_name == "Alpaca":
    dataset = ds['train']
    count = 0
    for data in tqdm.tqdm(dataset):
        start = time.time()
        instruction = data['instruction']
        input = data['input']
        output = data['output']
        tokens = tokenizer(instruction + '\n' + input, return_tensors='pt')
        generated_ids = model.generate(
            tokens.input_ids.to('cuda'),
            max_new_tokens=len(output)+1,
            do_sample=False
        )
        result = tokenizer.decode(generated_ids[0].tolist())
        with open("./tests/output/Alpaca", mode='a') as f:
            print("instruction:", instruction, "", "input:", input, "", "output:", result, "", "ref_output:", output, "",
                    str(time.time()-start), "------------------------------\n", sep='\n', file=f)

        count += 1
        if count >= 20:
            pass


print('峰值内存使用：{} Bytes'.format(torch.cuda.max_memory_allocated()))
print('缓存内存峰值：{} Bytes'.format(torch.cuda.max_memory_cached()))
