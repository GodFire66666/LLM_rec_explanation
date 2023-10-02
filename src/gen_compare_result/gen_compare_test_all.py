import sys
import json
import os

import fire
# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel, TextStreamer

# from utils.prompter import Prompter
import bitsandbytes as bnb

import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())

load_8bit = False
model_name = 'SFT'

if model_name == 'LLaMA2':
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    model_path = 'meta-llama/Llama-2-7b-chat-hf'
elif model_name == 'ChatGLM2':
    base_model = 'THUDM/chatglm2-6b'
    model_path = 'THUDM/chatglm2-6b'
elif model_name == 'SFT':
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    model_path = './model_SFT_discriminator/'
    lora_weights = model_path
if torch.cuda.is_available():
    device = "cuda"

if model_name in ['SFT']:
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"using lora {lora_weights}")
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

import json
import os.path as osp
from typing import Union

def get_instruction(instruction, exp1, exp2):
    prompt = "-"*20 + "Instruction" + "-"*20 + "\n"
    prompt += instruction + "\n"
    prompt += "-"*20 + "Explanation 1" + "-"*20 + "\n"
    prompt += exp1 + "\n"
    prompt += "-"*20 + "Explanation 2" + "-"*20 + "\n"
    prompt += exp2 + "\n"
    prompt += "-"*53 + "\n"

    return prompt

class Prompter_LLaMA2_discriminator(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, verbose: bool = False):
        super().__init__()

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
    ) -> str:        
        if input:
            prompt = instruction + input
        else:
            prompt = instruction
        system_message = "You are a discriminator that judges whether the explainability of the recommendation system is good or bad. You should judge which of the 2 interpretability opinions generated based on the following Instruction is better. Return 1 if you think the first one is better, and 2 if you think the second one is better. Only the number 1 or 2 should be returned. Do not return any other characters."
        prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt}Based on the above instructions, decide which explanation better explains why the recommendation system recommends this item to the customer.Please return 1 or 2 to show your choice. Only return 1 or 2. Do not return any other information. [/INST]'''
        
        return prompt_template

    def get_response(self, output: str) -> str:
        return output.split('[/INST]')[-1].strip(tokenizer.eos_token).strip()

prompter = Prompter_LLaMA2_discriminator()

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=4096,
    **kwargs,
):
    
    streamer = TextStreamer(tokenizer)
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            # streamer=streamer,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return instruction, prompter.get_response(output)


import pandas as pd
rec_model = 'LightGCN'

data_path = f'../gen_exp_5_model/results/all/{rec_model}'
data_path_steam = osp.join(data_path, 'steam.csv')
data_path_ml = osp.join(data_path, 'ml-100k.csv')
data_path_mind = osp.join(data_path, 'mind_small_dev.csv')

data_steam = pd.read_csv(data_path_steam)
data_ml = pd.read_csv(data_path_ml)
data_mind = pd.read_csv(data_path_mind)


def get_instruction_model(df, idx, model1, model2):
    data = df.iloc[idx]
    instruction = data['instruction']
    exp1 = data[model1]
    exp2 = data[model2]
    instruction = get_instruction(instruction, exp1, exp2)
    return instruction

from tqdm import tqdm
import random

df_all = pd.DataFrame(columns=['index', 'dataset', 'model1', 'model2', 'compare'])
index = 0


# for dataset in ['steam', 'ml-100k', 'mind_small_dev']:
for dataset in ['steam']:
    if dataset == 'steam':
        df = data_steam
    elif dataset == 'ml-100k':
        df = data_ml
    elif dataset == 'mind_small_dev':
        df = data_mind
    print(f'##############{dataset}##############')
    for a, b in [('LLaMA2', 'ChatGLM2'),
                           ('LLaMA2', 'GPT3.5'),
                           ('LLaMA2', 'GPT4'),
                           ('LLaMA2', 'LLaMA2-SFT'),
                           ('ChatGLM2', 'GPT3.5'),
                           ('ChatGLM2', 'GPT4'),
                           ('ChatGLM2', 'LLaMA2-SFT'),
                           ('GPT3.5', 'GPT4'),
                           ('GPT3.5', 'LLaMA2-SFT'),
                           ('GPT4', 'LLaMA2-SFT')]:
        print(a, b)
        for i in tqdm(range(df.shape[0])):
            # model1, model2 = random.sample([a, b], 2)
            model1, model2 = a, b
            instruction = get_instruction_model(df, i, model1, model2)
            instruction, response = evaluate(instruction)
            # print(response)
            # break
            df_all.loc[index] = [i, dataset, model1, model2, response]
            index += 1

if not os.path.exists(f'results/{rec_model}'):
    os.mkdir(f'results/{rec_model}')
df_all.to_csv(f'results/{rec_model}/compare_all.csv', index=False)

