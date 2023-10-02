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
    model_path = './model_SFT'
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


class Prompter_LLaMA2(object):
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
        system_message = "Please give a proper response to the instruction. Do not say 'I don't know."
        prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]'''
        
        return prompt_template

    def get_response(self, output: str) -> str:
        return output.split('[/INST]')[-1].strip(tokenizer.eos_token).strip()
    
class Prompter_ChatGLM2(object):
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
        
        prompt_template = prompt 
        return prompt_template

    def get_response(self, output: str) -> str:
        return output.strip(tokenizer.eos_token).strip()

prompter = Prompter_LLaMA2()

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

instruction = 'hello'
instruction, response = evaluate(instruction)
print(instruction)
print(response)

rec_model = 'LightGCN'
import pandas as pd
from tqdm import tqdm
for dataset_name in ['ml-100k', 'mind_small_dev','steam']:
    instruction_path = f'../gen_exp_4_model/results/{rec_model}/LLaMA2/{dataset_name}/results.csv'
    instruction_df = pd.read_csv(instruction_path)
    instruction_df

    def get_instruction(idx):
        instruction = instruction_df.iloc[idx]['instruction']
        return instruction

    result_save_path = f'./results/SFT/{rec_model}/{dataset_name}/'
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    df_result = pd.DataFrame(columns=['instruction', 'response'])
    # i = 0
    with open(result_save_path + 'results.csv', 'w', encoding='UTF-8') as f:
        for idx in tqdm(range(instruction_df.shape[0])):
            instruction, response = evaluate(get_instruction(idx))
            df_result = df_result.append({'instruction': instruction, 'response': response}, ignore_index=True)
            # print(instruction)
            # print(response)
            # break
        df_result.to_csv(f, index=False)
        f.close()


